import os
import glob
import math
from typing import List
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torchvision import models, transforms

# ====== 配置区 ======

#待检索数据集地址
SEARCH_DIR   = r"C:\Users\pc\Desktop\IEEE JS\GIPD\datasets"

# 与目标图像最相似的K个样本
K_NEIGHBORS  = 3
IMAGE_EXTS   = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp"]

# 一次送进神经网络的图片数量，可按显存调大/调小，我的显存是24GB，选择了一次送16张图片
BATCH_SIZE   = 16
# ====================

def list_images(folder: str, exts: List[str], target_image_to_exclude: str) -> List[str]:
    """在指定文件夹及其所有子目录中列出所有符合给定扩展名的图像文件路径，并排除目标图像本身。"""
    paths = []
    for ext in exts:
        # 使用 recursive=True 和 ** 通配符来递归搜索
        pattern = os.path.join(folder, '**', ext)
        paths.extend(glob.glob(pattern, recursive=True))
    # 去掉目标图像自身
    target_norm = os.path.normcase(os.path.abspath(target_image_to_exclude))
    paths = [p for p in paths if os.path.normcase(os.path.abspath(p)) != target_norm]
    return sorted(paths)

def load_feature_extractor(device):
    # 使用 torchvision 的 ResNet50 作为特征提取器（ImageNet 预训练）
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    # 去掉最后的分类层，保留到全局平均池化的 2048 维特征
    model.fc = nn.Identity()
    model.eval().to(device)

    # 对应预训练权重的官方预处理
    preprocess = weights.transforms()
    return model, preprocess

@torch.no_grad()
def extract_feature(img_path, model, preprocess, device):
    # 加载图像并做 EXIF 方向校正
    img = Image.open(img_path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    tensor = preprocess(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]
    feat = model(tensor)  # [1, 2048]
    feat = torch.nn.functional.normalize(feat, p=2, dim=1)  # L2 归一化，便于用余弦相似度
    return feat.squeeze(0).cpu().numpy()  # [2048]

def cosine_sim(a, b):
    # 输入已 L2 归一化，这里直接点积即可
    return float(np.dot(a, b))

def batched_extract(paths, model, preprocess, device, batch_size=16):
    feats = []
    with torch.no_grad():
        batch_imgs = []
        batch_idx = []
        for i, p in enumerate(paths):
            img = Image.open(p).convert("RGB")
            img = ImageOps.exif_transpose(img)
            batch_imgs.append(preprocess(img))
            batch_idx.append(i)
            if len(batch_imgs) == batch_size or i == len(paths) - 1:
                batch = torch.stack(batch_imgs, dim=0).to(device)
                f = model(batch)                           # [B, 2048]
                f = torch.nn.functional.normalize(f, p=2, dim=1)
                feats.append(f.cpu().numpy())              # [B, 2048]
                batch_imgs = []
                batch_idx = []
    if feats:
        return np.concatenate(feats, axis=0)
    else:
        return np.zeros((0, 2048), dtype=np.float32)

def find_k_nearest_neighbors(target_image_path: str, search_dir: str, k_neighbors: int, image_exts: List[str], batch_size: int = 16) -> List[str]:
    """
    根据 K-近邻算法查找与目标图像最相似的 K 个图像。

    Args:
        target_image_path: 目标图像的路径。
        search_dir: 待检索数据集的目录。
        k_neighbors: 与目标图像最相似的 K 个样本数量。
        image_exts: 图像文件扩展名列表。
        batch_size: 一次送进神经网络的图片数量。

    Returns:
        K 个最相似图像的路径列表。
    """
    # 检查路径
    if not os.path.isfile(target_image_path):
        raise FileNotFoundError(f"目标图像不存在：{target_image_path}")
    if not os.path.isdir(search_dir):
        raise NotADirectoryError(f"搜索目录不存在：{search_dir}")

    # 收集候选图像
    candidates = list_images(search_dir, image_exts, target_image_to_exclude=target_image_path)

    if len(candidates) == 0:
        print("未在目录中找到候选图像（或只剩目标图像本身）。")
        return []

    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 模型与预处理
    model, preprocess = load_feature_extractor(device)

    # 提取目标特征
    print("Extracting target feature...")
    target_feat = extract_feature(target_image_path, model, preprocess, device)

    # 批量提取候选特征
    print(f"Extracting features for {len(candidates)} candidate images...")
    cand_feats = batched_extract(candidates, model, preprocess, device, batch_size)

    # 计算相似度
    print("Computing cosine similarities...")
    sims = np.dot(cand_feats, target_feat)  # 因为都已 L2 归一化，点积即余弦相似
    # 取 Top-K
    topk_idx = np.argsort(-sims)[:k_neighbors]
    topk_paths = [candidates[i] for i in topk_idx]

    # 输出结果 (可选，为了调试)
    print("\n=== 最近邻样本（Top-{}） ===".format(k_neighbors))
    for rank, i in enumerate(topk_idx, start=1):
        print(f"{rank}. {candidates[i]} | cosine_sim={sims[i]:.4f}")
    
    return topk_paths

if __name__ == "__main__":
    # 仅在直接运行时执行原始 main 函数的逻辑，但调用新的函数
    # ====== 配置区 ======
    # 仅用于独立测试
    TARGET_IMAGE = r"C:\Users\pc\Desktop\IEEE JS\GIPD\datasets\AC1.png"
    SEARCH_DIR   = r"C:\Users\pc\Desktop\IEEE JS\GIPD\datasets"
    # 与目标图像最相似的K个样本
    K_NEIGHBORS  = 3
    IMAGE_EXTS   = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp"]
    # 一次送进神经网络的图片数量，可按显存调大/调小，我的显存是24GB，选择了一次送16张图片
    BATCH_SIZE   = 16
    # ====================
    
    similar_images = find_k_nearest_neighbors(TARGET_IMAGE, SEARCH_DIR, K_NEIGHBORS, IMAGE_EXTS, BATCH_SIZE)
    print("\nSimilar images found:", similar_images)
