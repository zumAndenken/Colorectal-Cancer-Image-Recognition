"""
卫星图像地块用途分类脚本

功能概述：
- 遍历给定的数据集目录，读取图片文件。
- 从文件名中解析出真实的分类标签（例如 'AC' -> 'active'）。
- 模型会输出预测类别、原因和置信度。
- 支持并发处理，将结果汇总写入 Excel，并生成分类评估指标（混淆矩阵、精确率、召回率等）。

关键参数：
- MAX_WORKERS：并发请求数量。并发越大，利用 GPU 越充分，但显存占用更高。
- MODEL: 使用的模型名称。
- NUM_EXAMPLES_PER_CLASS：示例图片数量
"""

import os
import re
import time
import base64
import random
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

# 导入 KNN 模块
from KNN import find_k_nearest_neighbors, IMAGE_EXTS, BATCH_SIZE as KNN_BATCH_SIZE

# ===== 固定配置 =====
ROOT_DIR = r"/gz-data/GIPD"
OUTPUT_FILE = r"/gz-data/qwen2.5vl_72b.xlsx"
OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen2.5vl:72b"  # 使用的模型名称

# 并发
MAX_WORKERS = 1  # 并发请求数

# 需要分析的文件扩展名
ALLOWED_EXTS = [".png", ".jpg", ".jpeg"]

# 文件选择控制（0 表示不限）
MAX_FILES = 0  # 全局最多处理的文件数
NUM_EXAMPLES_PER_CLASS = 9 # 用作 few-shot 示例的图片数量 (随机)

# KNN 相关配置
USE_DYNAMIC_KNN_EXAMPLES = False # 是否为每个待分析图片动态查找 KNN 示例
KNN_SEARCH_DIR = ROOT_DIR # KNN 搜索目录与根目录保持一致
K_NEIGHBORS_FOR_KNN = 0 # KNN 查找的近邻数量
# ===========================


class ImageClassifier:
    """图像分类器

    通过 HTTP 调用本地 Ollama 服务，对输入的卫星图像进行地块用途分类。
    """
    def __init__(
        self,
        ollama_url: str = OLLAMA_URL,
        model_name: str = MODEL,
        temperature: float = 0.5,
        request_timeout: int = 60,
        request_retries: int = 3,
        max_workers: int = 1,
        num_examples: int = NUM_EXAMPLES_PER_CLASS,
        use_dynamic_knn: bool = USE_DYNAMIC_KNN_EXAMPLES, # 新增参数
        knn_search_dir: Optional[str] = None, # 新增参数
        knn_k_neighbors: int = K_NEIGHBORS_FOR_KNN, # 新增参数
    ):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.temperature = float(temperature)
        self.request_timeout = int(request_timeout)
        self.request_retries = int(request_retries)
        self.max_workers = max(1, int(max_workers))

        self.session = requests.Session()

        self.land_use_classes = ["active", "brownfield", "construction"]
        
        # 标签映射：文件名 -> 标准类别
        self.label_map = {
            "AC": "active",
            "BR": "brownfield",
            "CO": "construction",
        }

        # 指导模型进行分类的系统提示词
        self.system_prompt = (
            "You are a satellite image analysis expert. Your task is to classify the given satellite image tile. "
            "Analyze what you see in the image and output only the single most likely land use type from the following list:\n"
            "- active: The property is actively used for industrial or commercial purposes.\n"
            "- brownfield: The property is abandoned, unused, or underutilized.\n"
            "- construction: The property is an active construction site.\n\n"
            "Your output should be only the land use type (e.g., active, brownfield, construction)."
        )
        
        self.use_dynamic_knn = use_dynamic_knn
        self.knn_search_dir = knn_search_dir if knn_search_dir else ROOT_DIR # 如果未指定，则使用 ROOT_DIR
        self.knn_k_neighbors = knn_k_neighbors
        self.num_examples = num_examples

    def _get_image_base64(self, path: str) -> Optional[str]:
        """读取图片文件并返回 Base64 编码的字符串。"""
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"Error reading or encoding image {path}: {e}")
            return None

    def _load_example_images(self, target_image_path: Optional[str] = None) -> Dict[str, List[str]]:
        """
        为给定的目标图像加载 few-shot 示例。
        如果 self.use_dynamic_knn 为 True，则使用 KNN 算法查找相似图片作为示例。
        否则，从整个数据集中随机采样图片作为示例。
        """
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
                    long_label = self.label_map.get(category_name, "unknown")

                    if long_label != "unknown":
                        b64_content = self._get_image_base64(img_path)
                        if b64_content:
                            if long_label not in example_images:
                                example_images[long_label] = []
                            example_images[long_label].append(b64_content)
                            print(f"  - Loaded KNN example for '{long_label}': {os.path.basename(img_path)}")
                    else:
                        print(f"Warning: Could not determine class for KNN example: {img_path}")
            
            except Exception as e:
                raise RuntimeError(f"Error during KNN example loading for '{target_image_path}': {e}") from e

        elif self.num_examples > 0:
            print(f"Loading {self.num_examples} random example images from the entire dataset for few-shot learning...")
            
            all_available_example_files = []
            for short_label in self.label_map.keys():
                category_dir = os.path.join(ROOT_DIR, short_label)
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
                    long_label = self.label_map.get(category_name, "unknown")

                    if long_label != "unknown":
                        b64_content = self._get_image_base64(file_path)
                        if b64_content:
                            if long_label not in example_images:
                                example_images[long_label] = []
                            example_images[long_label].append(b64_content)
                            print(f"  - Loaded '{long_label}' random example: {os.path.basename(file_path)}")
                    else:
                        print(f"Warning: Could not determine class for random example: {file_path}")
            else:
                print("Warning: No valid image files found for random sampling in the entire dataset.")
        
        return example_images

    def call_ollama_vision(self, image_path: str, use_knn: bool) -> str:
        """调用 Ollama /api/generate 接口（支持图片），返回纯文本响应。"""
        image_b64 = self._get_image_base64(image_path)
        if not image_b64:
            return ""

        # 始终传递 target_image_path 以便在需要时排除自身
        example_images_b64 = self._load_example_images(target_image_path=image_path)

        # 构建 few-shot prompt
        messages = []

        # System prompt
        messages.append({"role": "system", "content": self.system_prompt})

        # Few-shot examples in a question-and-answer format
        for label, b64_img_list in example_images_b64.items():
            for b64_img in b64_img_list:
                # User asks to classify an example image
                messages.append({
                    "role": "user",
                    "content": f"This is an example of a '{label}'.",
                    "images": [b64_img]
                })
                # Assistant provides the correct label
                messages.append({
                    "role": "assistant",
                    "content": label
                })

        # Final image to analyze
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
                # For /api/chat, the response is in data["message"]["content"]
                return data.get("message", {}).get("content", "")
            except requests.exceptions.RequestException as e:
                if attempt < self.request_retries:
                    wait = min(2**attempt, 8)
                    print(f"调用Ollama失败(第 {attempt}/{self.request_retries} 次)，重试前等待 {wait}s：{e}")
                    time.sleep(wait)
                else:
                    print(f"调用Ollama失败(最终失败)：{e}")
        return ""

    def parse_classification_response(self, response: str) -> Dict[str, any]:
        """从模型 JSON 输出中解析分类结果、置信度。"""
        import json

        results = { "PredictedClass": "unknown" }

        if not response:
            return results

        predicted_class = "unknown"
        
        # 尝试从明确的格式中提取类别，例如 "**active**" 或独立一行
        # 匹配 "**CLASS_NAME**"
        match_bold = re.search(r'\*\*(' + '|'.join(re.escape(cls) for cls in self.land_use_classes) + r')\*\*', response, re.IGNORECASE)
        if match_bold:
            # 找到匹配的类别，并确保大小写与原始类别列表一致
            for cls in self.land_use_classes:
                if cls.lower() == match_bold.group(1).lower():
                    predicted_class = cls
                    break
        else:
            # 如果没有找到粗体匹配，尝试匹配独立一行或以类别名称开头的行
            match_line = re.search(r'^\s*(' + '|'.join(re.escape(cls) for cls in self.land_use_classes) + r')\s*$', response, re.MULTILINE | re.IGNORECASE)
            if match_line:
                for cls in self.land_use_classes:
                    if cls.lower() == match_line.group(1).lower():
                        predicted_class = cls
                        break
            else:
                # 如果以上都失败，回退到遍历所有类别进行模糊匹配
                for cls in self.land_use_classes:
                    if re.search(r'\b' + re.escape(cls) + r'\b', response, re.IGNORECASE):
                        predicted_class = cls
                        break
        
        results["PredictedClass"] = predicted_class

        return results

    def compute_results_metrics(self, df_output: pd.DataFrame) -> pd.DataFrame:
        """基于结果表计算分类指标。"""
        if df_output is None or df_output.empty or "ActualClass" not in df_output or "PredictedClass" not in df_output:
            print("无法计算指标：输入数据不完整。")
            return pd.DataFrame(), pd.DataFrame()

        y_true = df_output["ActualClass"]
        y_pred = df_output["PredictedClass"]
        labels = self.land_use_classes

        # 总体准确率（基于 dominant_class）
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nOverall Accuracy (based on dominant class): {accuracy:.4f}")

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
        root_dir = os.path.abspath(root_dir)
        
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # 排除 .ipynb_checkpoints 目录
            if '.ipynb_checkpoints' in dirnames:
                dirnames.remove('.ipynb_checkpoints')

            for filename in filenames:
                if not filename.lower().endswith(allowed_exts):
                    continue

                file_path = os.path.join(dirpath, filename)
                
                # 从父目录名获取标签
                # 例如: E:\TEST\SatelliteDatasets\AC -> AC
                category_name = os.path.basename(dirpath).upper()
                actual_label = self.label_map.get(category_name, "unknown")

                if actual_label == "unknown":
                    # 如果根目录直接包含图片，则跳过
                    if dirpath == root_dir:
                        continue
                    print(f"无法从目录名 '{category_name}' 解析标签，跳过: {file_path}")
                    continue
                
                items.append((filename, file_path, actual_label))
        return items

    def _analyze_one_file(self, filename: str, file_path: str, actual_label: str) -> Optional[Dict[str, any]]:
        """读取并分析单个图片文件，返回结果记录。"""
        start_time = time.time()
        response_text = self.call_ollama_vision(file_path, use_knn=self.use_dynamic_knn)
        analysis_time = time.time() - start_time

        parsed_results = self.parse_classification_response(response_text)

        is_correct = "√" if actual_label == parsed_results["PredictedClass"] else "×"
        
        row = {
            "FileID": filename,
            "ActualClass": actual_label,
            "PredictedClass": parsed_results["PredictedClass"],
            "IsCorrect": is_correct,
            "AnalysisTime": round(analysis_time, 2),
        }
        print(f"完成: {filename} (耗时: {analysis_time:.2f}s) -> {row['PredictedClass']}")
        return row

    def process_folder(
        self,
        root_dir: str,
        output_file: str,
        allowed_exts: List[str],
        max_files: int = 0,
    ):
        """目录模式主流程：收集、分析并将结果写入 Excel。"""
        if not os.path.isdir(root_dir):
            print(f"目录不存在: {root_dir}")
            return

        allowed_exts_t = tuple(x.lower() for x in allowed_exts)
        items = self._gather_files(root_dir, allowed_exts_t)

        if max_files and max_files > 0:
            items = items[:max_files]

        if not items:
            print("未发现可处理的文件")
            return

        print(f"将处理 {len(items)} 个文件...")

        results = []
        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                futs = [ex.submit(self._analyze_one_file, fname, fpath, label) for fname, fpath, label in items]
                for fut in as_completed(futs):
                    if r := fut.result():
                        results.append(r)
        else:
            for fname, fpath, label in items:
                if r := self._analyze_one_file(fname, fpath, label):
                    results.append(r)

        if not results:
            print("没有生成任何结果。")
            return

        df_output = pd.DataFrame(results)

        # 尝试读取现有文件，如果存在则合并数据
        try:
            # 检查文件是否存在且非空
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                df_existing = pd.read_excel(output_file, sheet_name='Raw_Results')
                df_combined = pd.concat([df_existing, df_output], ignore_index=True)
            else:
                df_combined = df_output
        except Exception as e:
            print(f"读取现有 Excel 文件失败: {e}。将只保存当前结果。")
            df_combined = df_output

        # 基于合并后的所有数据计算总体指标
        print("\nComputing overall metrics for all combined data...")
        metrics_df, cm_df = self.compute_results_metrics(df_combined)

        # 将所有内容写入 Excel 文件（覆盖模式，但包含追加的数据）
        try:
            was_existing = os.path.exists(output_file)
            with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
                df_combined.to_excel(writer, sheet_name="Raw_Results", index=False)
                metrics_df.to_excel(writer, sheet_name="Per_Class_Metrics", index=False)
                cm_df.to_excel(writer, sheet_name="Confusion_Matrix")
            
            if was_existing:
                print(f"\n结果已追加，评估指标已更新: {output_file}")
            else:
                print(f"\n结果已保存到新文件: {output_file}")

        except Exception as e:
            print(f"保存结果到 Excel 文件时出错: {e}")


def main():
    """程序入口：构造分类器并按配置运行目录模式。"""
    analyzer = ImageClassifier(
        ollama_url=OLLAMA_URL,
        model_name=MODEL,
        max_workers=MAX_WORKERS,
        num_examples=NUM_EXAMPLES_PER_CLASS,
        use_dynamic_knn=USE_DYNAMIC_KNN_EXAMPLES, # 传入 KNN 开关
        knn_search_dir=KNN_SEARCH_DIR, # 传入 KNN 搜索目录
        knn_k_neighbors=K_NEIGHBORS_FOR_KNN, # 传入 KNN 近邻数量
    )
    analyzer.process_folder(
        root_dir=ROOT_DIR,
        output_file=OUTPUT_FILE,
        allowed_exts=ALLOWED_EXTS,
        max_files=MAX_FILES,
    )


if __name__ == "__main__":
    main()