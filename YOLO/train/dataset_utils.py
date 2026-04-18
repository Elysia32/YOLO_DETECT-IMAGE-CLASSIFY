# -*- coding: utf-8 -*-
"""
数据集分割工具函数
"""
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split


SUPPORTED_IMG_EXTS = ['.jpg', '.jpeg', '.png']


def find_image_file(stem, img_dir, exts):
    """根据标签文件名查找对应的图片文件"""
    for ext in exts:
        img_path = os.path.join(img_dir, stem + ext)
        if os.path.exists(img_path):
            return img_path, ext
    return None, None


def split_dataset(imgpath: str, txtpath: str, output_dir: str,
                  test_ratio: float, val_ratio: float) -> dict:
    """
    分割数据集为 train/val/test 三部分

    Args:
        imgpath: 原始图片目录
        txtpath: 原始标签目录
        output_dir: 输出目录
        test_ratio: 测试集比例 (0-1)
        val_ratio: 验证集在剩余数据中的比例 (0-1)

    Returns:
        dict: 包含统计信息的字典 {train_count, val_count, test_count, missing_count}
    """
    BASE_DIR = Path(output_dir)

    folders = {
        'train': {'img': BASE_DIR / 'images' / 'train', 'lbl': BASE_DIR / 'labels' / 'train'},
        'val':   {'img': BASE_DIR / 'images' / 'val',   'lbl': BASE_DIR / 'labels' / 'val'},
        'test':  {'img': BASE_DIR / 'images' / 'test',  'lbl': BASE_DIR / 'labels' / 'test'},
    }

    # 创建目录结构
    for f in folders.values():
        f['img'].mkdir(parents=True, exist_ok=True)
        f['lbl'].mkdir(parents=True, exist_ok=True)

    # 获取所有标签文件
    txt_files = [f for f in os.listdir(txtpath) if f.endswith('.txt')]

    # 配对标签和图片
    paired_files = []
    missing_imgs = []

    for txt_file in txt_files:
        stem = txt_file[:-4]
        img_file, _ = find_image_file(stem, imgpath, SUPPORTED_IMG_EXTS)
        if img_file:
            paired_files.append((txt_file, img_file))
        else:
            missing_imgs.append(txt_file)

    # 执行划分
    txt_list = [pair[0] for pair in paired_files]
    trainval_txt, test_txt = train_test_split(txt_list, test_size=test_ratio, random_state=42)
    relative_val_ratio = val_ratio / (1 - test_ratio)
    train_txt, val_txt = train_test_split(trainval_txt, test_size=relative_val_ratio, random_state=42)

    split_map = {}
    for t in train_txt:
        split_map[t] = 'train'
    for v in val_txt:
        split_map[v] = 'val'
    for te in test_txt:
        split_map[te] = 'test'

    def copy_file_pair(txt_file, img_file, split):
        dst_img = folders[split]['img'] / os.path.basename(img_file)
        dst_lbl = folders[split]['lbl'] / txt_file
        shutil.copy(img_file, dst_img)
        shutil.copy(os.path.join(txtpath, txt_file), dst_lbl)

    # 复制文件
    for txt_file, img_file in paired_files:
        split = split_map[txt_file]
        copy_file_pair(txt_file, img_file, split)

    return {
        'train_count': len(train_txt),
        'val_count': len(val_txt),
        'test_count': len(test_txt),
        'missing_count': len(missing_imgs),
        'total_files': len(paired_files)
    }


def scan_dataset_stats(images_dir: str) -> dict:
    """
    扫描数据集统计信息

    Returns:
        dict: {train_count, val_count, classes}
    """
    images_path = Path(images_dir)
    stats = {'train_count': 0, 'val_count': 0, 'classes': []}

    # 统计训练集图片数
    train_img_dir = images_path / 'train' / 'images'
    if train_img_dir.exists():
        stats['train_count'] = len([f for f in train_img_dir.iterdir() if f.suffix.lower() in SUPPORTED_IMG_EXTS])

    # 统计验证集图片数
    val_img_dir = images_path / 'val' / 'images'
    if val_img_dir.exists():
        stats['val_count'] = len([f for f in val_img_dir.iterdir() if f.suffix.lower() in SUPPORTED_IMG_EXTS])

    # 尝试从标签文件中提取类别
    labels_dir = images_path / 'train' / 'labels'
    if labels_dir.exists():
        class_set = set()
        for txt_file in labels_dir.glob('*.txt'):
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = parts[0]
                        class_set.add(class_id)
        stats['classes'] = sorted(list(class_set))

    return stats


def extract_classes_from_labels(labels_dir: str) -> list:
    """
    从标签文件中提取唯一的类别名称（第一列）
    """
    labels_path = Path(labels_dir)
    class_set = set()

    for txt_file in labels_path.glob('*.txt'):
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_set.add(parts[0])

    return sorted(list(class_set))


def generate_yaml_content(train_path: str, val_path: str,
                          nc: int, names: list,
                          auto_path: bool = True) -> str:
    """
    生成 YAML 配置文件内容

    Args:
        train_path: 训练集图片路径
        val_path: 验证集图片路径
        nc: 类别数量
        names: 类别名称列表
        auto_path: 是否使用相对路径格式

    Returns:
        str: YAML 格式字符串
    """
    # 确保路径使用反斜杠 (Windows 风格)
    train_path = train_path.replace('/', '\\')
    val_path = val_path.replace('/', '\\')

    # 将 names 列表格式化为 YAML 数组形式 [xxx, xxx]
    lines = [
        f"train: {train_path}",
        f"val: {val_path}",
        "",
        f"nc: {nc}",
        "",
        "names: [" + ", ".join([f"'{name}'" for name in names]) + "]"
    ]

    return "\n".join(lines)
