"""
MHIST 数据集 Zero-Shot/Few-Shot 分类脚本

功能概述：
- 遍历 MHIST 数据集目录，读取组织病理学图像文件。
- 从目录名中解析出真实的分类标签 (HP 或 SSA)。
- 支持 Zero-Shot 或 Few-Shot (通过 KNN 动态查找示例) 模式。
- 脚本根据模型返回的文本响应解析出最终预测类别。
- 支持并发处理，将结果汇总写入 Excel，并生成分类评估指标（混淆矩阵、准确率等）。
"""

import os
import re
import time
import base64
import json
import random
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from sklearn.metrics import confusion_matrix, accuracy_score

# 导入 KNN 模块
from KNN import find_k_nearest_neighbors, IMAGE_EXTS, BATCH_SIZE as KNN_BATCH_SIZE

# ===== 固定配置 =====
ROOT_DIR = r"/gz-data/MHIST"  # MHIST 数据集根目录
OUTPUT_FILE = r"/gz-data/gemma3:12b.xlsx" # 输出文件路径
OLLAMA_URL = "http://localhost:11434"
MODEL = "gemma3:12b"  # 使用的模型名称

# 并发
MAX_WORKERS = 1  # 并发请求数

# 需要分析的文件扩展名
ALLOWED_EXTS = [".tif", ".png", ".jpg", ".jpeg"]

# 文件选择控制（0 表示不限）
MAX_FILES = 0  # 全局最多处理的文件数
SKIP_FILES_COUNT = 0 # 跳过前 N 个文件
NUM_EXAMPLE_CLASSES = 0 # 用作 few-shot 示例的图片数量 (从整个数据集中随机取)

# KNN 相关配置
USE_DYNAMIC_KNN_EXAMPLES = False # 是否为每个待分析图片动态查找 KNN 示例
KNN_SEARCH_DIR = ROOT_DIR # KNN 搜索目录与根目录保持一致
K_NEIGHBORS_FOR_KNN = 6 # KNN 查找的近邻数量
# ===========================


class MHISTClassifier:
    """MHIST 组织病理学图像分类器"""
    def __init__(
        self,
        ollama_url: str = OLLAMA_URL,
        model_name: str = MODEL,
        temperature: float = 0.5, # 对于分类任务，较低的温度通常更好
        request_timeout: int = 30,
        request_retries: int = 3,
        max_workers: int = 1,
        num_examples: int = NUM_EXAMPLE_CLASSES,
        use_dynamic_knn: bool = USE_DYNAMIC_KNN_EXAMPLES,
        knn_search_dir: Optional[str] = None,
        knn_k_neighbors: int = K_NEIGHBORS_FOR_KNN,
    ):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.temperature = float(temperature)
        self.request_timeout = int(request_timeout)
        self.request_retries = int(request_retries)
        self.max_workers = max(1, int(max_workers))
        self.session = requests.Session()

        self.tissue_classes = ["HP", "SSA"]
        self.label_map = {label: label for label in self.tissue_classes}

        self.system_prompt = (
            "You are an expert pathologist. Your task is to classify the given histopathology image tile from a colon polyp sample. "
            "Analyze what you see in the image,Based on your analysis, output only the single most likely tissue type from the following list:\n"
            "- HP: Hyperplastic Polyp\n"
            "- SSA: Sessile Serrated Adenoma\n\n"
            "Your output should be only the tissue type abbreviation."
        )

        self.num_examples = num_examples
        self.use_dynamic_knn = use_dynamic_knn
        self.knn_search_dir = knn_search_dir if knn_search_dir else ROOT_DIR
        self.knn_k_neighbors = knn_k_neighbors

    def _get_image_base64(self, path: str) -> Optional[str]:
        """读取图片文件并返回 Base64 编码的字符串。"""
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"Error reading or encoding image {path}: {e}")
            return None

    def _load_example_images(self, target_image_path: Optional[str] = None) -> Dict[str, List[str]]:
        """为给定的目标图像加载 few-shot 示例。"""
        example_images: Dict[str, List[str]] = {}
        
        if self.use_dynamic_knn and target_image_path:
            print(f"Using KNN to find {self.knn_k_neighbors} similar images for '{target_image_path}' as few-shot examples...")
            try:
                similar_image_paths = find_k_nearest_neighbors(
                    target_image_path=target_image_path,
                    search_dir=self.knn_search_dir,
                    k_neighbors=self.knn_k_neighbors,
                    image_exts=IMAGE_EXTS,
                    batch_size=KNN_BATCH_SIZE
                )
                
                if not similar_image_paths:
                    raise RuntimeError("KNN did not find any similar images.")

                for img_path in similar_image_paths:
                    category_name = os.path.basename(os.path.dirname(img_path)).upper()
                    label = self.label_map.get(category_name, "unknown")

                    if label != "unknown":
                        b64_content = self._get_image_base64(img_path)
                        if b64_content:
                            if label not in example_images:
                                example_images[label] = []
                            example_images[label].append(b64_content)
                            print(f"  - Loaded KNN example for '{label}': {os.path.basename(img_path)}")
                    else:
                        print(f"Warning: Could not determine class for KNN example: {img_path}")
            
            except Exception as e:
                raise RuntimeError(f"Error during KNN example loading for '{target_image_path}': {e}") from e

        elif self.num_examples > 0:
            print(f"Loading {self.num_examples} random example images from the entire dataset for few-shot learning...")
            
            all_available_example_files = []
            for label in self.tissue_classes:
                category_dir = os.path.join(ROOT_DIR, label)
                if not os.path.isdir(category_dir):
                    print(f"Warning: Example directory not found: {category_dir}")
                    continue
                
                all_files_in_dir = [
                    os.path.join(category_dir, f)
                    for f in os.listdir(category_dir)
                    if f.lower().endswith(tuple(ALLOWED_EXTS))
                ]
                all_available_example_files.extend(all_files_in_dir)

            # Exclude the current image if it happens to be in the example pool
            if target_image_path and target_image_path in all_available_example_files:
                all_available_example_files.remove(target_image_path)

            if all_available_example_files:
                num_to_sample = min(self.num_examples, len(all_available_example_files))
                sampled_files = random.sample(all_available_example_files, num_to_sample)
                
                for file_path in sampled_files:
                    category_name = os.path.basename(os.path.dirname(file_path)).upper()
                    label = self.label_map.get(category_name, "unknown")

                    if label != "unknown":
                        b64_content = self._get_image_base64(file_path)
                        if b64_content:
                            if label not in example_images:
                                example_images[label] = []
                            example_images[label].append(b64_content)
                            print(f"  - Loaded '{label}' random example: {os.path.basename(file_path)}")
                    else:
                        print(f"Warning: Could not determine class for random example: {file_path}")
            else:
                print("Warning: No valid image files found for random sampling in the entire dataset.")
        
        return example_images

    def call_ollama_vision(self, image_path: str) -> str:
        """调用 Ollama /api/chat 接口（支持图片），返回纯文本响应。"""
        image_b64 = self._get_image_base64(image_path)
        if not image_b64:
            return ""

        example_images_b64 = self._load_example_images(target_image_path=image_path)

        messages = [{"role": "system", "content": self.system_prompt}]

        for label, b64_img_list in example_images_b64.items():
            for b64_img in b64_img_list:
                messages.append({
                    "role": "user",
                    "content": f"This is an example of a '{label}'.",
                    "images": [b64_img]
                })
                messages.append({"role": "assistant", "content": label})

        messages.append({
            "role": "user",
            "content": "Now, classify this new image based on the examples and instructions provided.",
            "images": [image_b64]
        })

        url = f"{self.ollama_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": 40960,
            },
        }

        for attempt in range(1, self.request_retries + 1):
            try:
                resp = self.session.post(url, json=payload, timeout=self.request_timeout)
                resp.raise_for_status()
                data = resp.json()
                return data.get("message", {}).get("content", "")
            except requests.exceptions.RequestException as e:
                if attempt < self.request_retries:
                    wait = min(2**attempt, 8)
                    print(f"Ollama chat call failed (attempt {attempt}/{self.request_retries}), retrying in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    print(f"Ollama chat call failed (final attempt): {e}")
        return ""

    def parse_classification_response(self, response: str) -> Dict[str, any]:
        """从模型响应中解析分类结果。"""
        results = { "PredictedClass": "unknown" }

        if not response:
            return results

        predicted_class = "unknown"
        
        # 优先匹配独立一行的类别缩写，以提高准确性
        match_line = re.search(r'^\s*(' + '|'.join(re.escape(cls) for cls in self.tissue_classes) + r')\s*$', response, re.MULTILINE | re.IGNORECASE)
        if match_line:
            for cls in self.tissue_classes:
                if cls.lower() == match_line.group(1).lower():
                    predicted_class = cls
                    break
        else:
            # 如果没有找到独立行的匹配，回退到在整个响应中查找类别
            for cls in self.tissue_classes:
                if re.search(r'\b' + re.escape(cls) + r'\b', response, re.IGNORECASE):
                    predicted_class = cls
                    break
        
        results["PredictedClass"] = predicted_class
        return results

    def compute_results_metrics(self, df_output: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """基于结果表计算分类指标。"""
        if df_output is None or df_output.empty:
            print("Cannot compute metrics: input data is empty.")
            return pd.DataFrame(), pd.DataFrame()

        y_true = df_output["ActualClass"]
        y_pred = df_output["PredictedClass"]
        labels = self.tissue_classes

        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nOverall Accuracy: {accuracy:.4f}")

        metrics_summary = {"Overall_Accuracy": [accuracy]}
        metrics_df = pd.DataFrame(metrics_summary)

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=[f"Actual_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels])
        
        print("\nConfusion Matrix:")
        print(cm_df)

        return metrics_df, cm_df

    def _gather_files(self, root_dir: str, allowed_exts: Tuple[str, ...]) -> List[Tuple[str, str, str]]:
        """递归遍历根目录，收集文件并从父目录名解析标签。"""
        items = []
        for dirpath, _, filenames in os.walk(root_dir):
            category_name = os.path.basename(dirpath).upper()
            if category_name not in self.label_map:
                continue

            for filename in filenames:
                if not filename.lower().endswith(allowed_exts):
                    continue
                
                file_path = os.path.join(dirpath, filename)
                unique_id = os.path.relpath(file_path, root_dir).replace('\\', '/')
                actual_label = self.label_map[category_name]
                items.append((unique_id, file_path, actual_label))
        return items

    def _analyze_one_file(self, file_id: str, file_path: str, actual_label: str) -> Optional[Dict[str, any]]:
        """读取并分析单个图片文件，返回结果记录。"""
        
        retries = 0
        response_text = ""
        parsed_results = {"PredictedClass": "unknown"}
        start_time = time.time()

        while parsed_results["PredictedClass"] == "unknown":
            if retries > 0:
                print(f"Retrying classification for {file_id} (attempt {retries})...")
                time.sleep(1) # 每次重试前等待1秒

            response_text = self.call_ollama_vision(file_path)
            parsed_results = self.parse_classification_response(response_text)
            retries += 1

        analysis_time = time.time() - start_time

        print(f"\n--- AI Analysis for {file_id} ---")
        print(response_text)
        print("-----------------------------------\n")

        is_correct = "√" if actual_label == parsed_results["PredictedClass"] else "×"
        
        row = {
            "FileID": file_id,
            "ActualClass": actual_label,
            "PredictedClass": parsed_results["PredictedClass"],
            "IsCorrect": is_correct,
            "AnalysisTime": round(analysis_time, 2),
        }
            
        print(f"Finished: {file_id} (Time: {analysis_time:.2f}s) -> Pred: {row['PredictedClass']} (Actual: {actual_label}) (Retries: {retries - 1})")
        return row

    def process_folder(
        self,
        root_dir: str,
        output_file: str,
        allowed_exts: List[str],
        max_files: int = 0,
        skip_files_count: int = 0, # 新增参数
    ):
        """目录模式主流程：收集、分析并将结果写入 Excel。"""
        if not os.path.isdir(root_dir):
            print(f"Directory not found: {root_dir}")
            return
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        allowed_exts_t = tuple(x.lower() for x in allowed_exts)
        
        existing_df = pd.DataFrame()
        if os.path.exists(output_file):
            try:
                existing_df = pd.read_excel(output_file, sheet_name="Raw_Results")
                print(f"Loaded {len(existing_df)} existing records from {output_file}")
            except Exception as e:
                print(f"Could not read existing file, will create a new one. Error: {e}")
        
        all_files = self._gather_files(root_dir, allowed_exts_t)
        items_to_process = all_files

        # 应用跳过文件逻辑
        if skip_files_count > 0:
            print(f"Skipping the first {skip_files_count} files.")
            items_to_process = items_to_process[skip_files_count:]

        if max_files > 0:
            # 应用 max_files 限制
            items_to_process = items_to_process[:max_files]

        if not items_to_process:
            print("No files to process based on skip/max_files limits or empty directory.")
            # 如果没有新文件，但有旧数据，重新计算并更新指标
            if not existing_df.empty:
                print("\nRe-computing metrics for existing data...")
                metrics_df, cm_df = self.compute_results_metrics(existing_df)
                try:
                    # 使用追加模式 'a' 并替换工作表
                    with pd.ExcelWriter(output_file, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                        metrics_df.to_excel(writer, sheet_name="Metrics_Summary", index=False)
                        cm_df.to_excel(writer, sheet_name="Confusion_Matrix")
                    print(f"\nMetrics re-saved to: {output_file}")
                except FileNotFoundError:
                    # 如果文件被意外删除，则用现有数据重新创建
                    with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
                        existing_df.to_excel(writer, sheet_name="Raw_Results", index=False)
                        metrics_df.to_excel(writer, sheet_name="Metrics_Summary", index=False)
                        cm_df.to_excel(writer, sheet_name="Confusion_Matrix")
                    print(f"\nMetrics saved to new file: {output_file}")
                except Exception as e:
                    print(f"Error saving metrics to Excel: {e}")
            return

        print(f"Processing {len(items_to_process)} new files...")

        new_results = []
        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                futs = [ex.submit(self._analyze_one_file, file_id, fpath, label) for file_id, fpath, label in items_to_process]
                for fut in as_completed(futs):
                    if r := fut.result():
                        new_results.append(r)
        else:
            for file_id, fpath, label in items_to_process:
                if r := self._analyze_one_file(file_id, fpath, label):
                    new_results.append(r)

        if not new_results:
            print("No new results were generated.")
            return

        new_df = pd.DataFrame(new_results)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)

        print("\nComputing overall metrics for all data...")
        metrics_df, cm_df = self.compute_results_metrics(combined_df)

        try:
            with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
                combined_df.to_excel(writer, sheet_name="Raw_Results", index=False)
                metrics_df.to_excel(writer, sheet_name="Metrics_Summary", index=False)
                cm_df.to_excel(writer, sheet_name="Confusion_Matrix")
            print(f"\nResults saved to: {output_file}")
        except Exception as e:
            print(f"Error saving results to Excel: {e}")


def main():
    """程序入口"""
    analyzer = MHISTClassifier(
        ollama_url=OLLAMA_URL,
        model_name=MODEL,
        max_workers=MAX_WORKERS,
        num_examples=NUM_EXAMPLE_CLASSES,
        use_dynamic_knn=USE_DYNAMIC_KNN_EXAMPLES,
        knn_search_dir=KNN_SEARCH_DIR,
        knn_k_neighbors=K_NEIGHBORS_FOR_KNN,
    )
    analyzer.process_folder(
        root_dir=ROOT_DIR,
        output_file=OUTPUT_FILE,
        allowed_exts=ALLOWED_EXTS,
        max_files=MAX_FILES,
        skip_files_count=SKIP_FILES_COUNT, # 传递新的参数
    )


if __name__ == "__main__":
    main()