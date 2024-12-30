import os
import shutil
from pathlib import Path

def prepare_food101_dataset(data_dir: str, output_dir: str):
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    train_output_dir = output_dir / 'train'
    test_output_dir = output_dir / 'test'
    
    
    train_output_dir.mkdir(parents=True, exist_ok=True)
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    
    with open(data_dir / 'meta' / 'train.txt', 'r') as f:
        train_files = f.read().splitlines()
    
    with open(data_dir / 'meta' / 'test.txt', 'r') as f:
        test_files = f.read().splitlines()
    
    
    for file in train_files:
        src = data_dir / 'images' / (file + '.jpg')
        dst = train_output_dir / (file + '.jpg')
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
    
    
    for file in test_files:
        src = data_dir / 'images' / (file + '.jpg')
        dst = test_output_dir / (file + '.jpg')
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
    
    print(f"Food-101: {output_dir} completed.")

# 예제 사용법
data_dir = '/home/hice1/kpk6/scratch/Datasets/food-101/'  # 압축을 해제한 폴더 경로
output_dir = '/home/hice1/kpk6/scratch/Datasets/food-101_tmp/'  # 준비된 데이터셋을 저장할 폴더 경로

prepare_food101_dataset(data_dir, output_dir)
