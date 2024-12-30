# import os
# import shutil
# from pathlib import Path
# from sklearn.model_selection import train_test_split

# def prepare_caltech101_dataset(data_dir: str, output_dir: str, test_size: float = 0.2):
#     # 경로 설정
#     data_dir = Path(data_dir)
#     output_dir = Path(output_dir)
#     train_output_dir = output_dir / 'train'
#     test_output_dir = output_dir / 'test'
    
#     # 출력 디렉토리 생성
#     train_output_dir.mkdir(parents=True, exist_ok=True)
#     test_output_dir.mkdir(parents=True, exist_ok=True)
    
#     # 각 클래스에 대해 train/test 파일 복사
#     for class_dir in data_dir.iterdir():
#         if class_dir.is_dir():
#             images = list(class_dir.glob('*.jpg'))
#             print(images)
#             train_images, test_images = train_test_split(images, test_size=test_size, random_state=42)
            
#             # train 파일 복사
#             for img_path in train_images:
#                 dst = train_output_dir / class_dir.name / img_path.name
#                 dst.parent.mkdir(parents=True, exist_ok=True)
#                 shutil.copy(img_path, dst)
            
#             # test 파일 복사
#             for img_path in test_images:
#                 dst = test_output_dir / class_dir.name / img_path.name
#                 dst.parent.mkdir(parents=True, exist_ok=True)
#                 shutil.copy(img_path, dst)
    

# data_dir = '/data/sophia/caltech-101/101_ObjectCategories'  # 압축을 해제한 폴더 경로
# output_dir = '/data/sophia/caltech-101/101_ObjectCategories_tmp'  # 준비된 데이터셋을 저장할 폴더 경로

# prepare_caltech101_dataset(data_dir, output_dir)

import os
import shutil
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

def prepare_caltech101_dataset(data_dir: str, output_dir: str, test_size: float = 0.2):
    # 경로 설정
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    train_output_dir = output_dir / 'train'
    test_output_dir = output_dir / 'test'
    
    # 출력 디렉토리 생성
    train_output_dir.mkdir(parents=True, exist_ok=True)
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 각 클래스에 대해 train/test 파일 복사
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob('*.jpg'))
            valid_images = []
            
            # 이미지를 로드하고 변환
            for img_path in images:
                try:
                    with Image.open(img_path) as img:
                        img = img.convert('RGB')  # 이미지를 RGB로 변환
                        valid_images.append(img_path)
                except:
                    print(f"Deleting corrupted image: {img_path}")
                    #os.remove(img_path)
            
            if valid_images:
                train_images, test_images = train_test_split(valid_images, test_size=test_size, random_state=42)
                
                # train 파일 복사
                for img_path in train_images:
                    dst = train_output_dir / class_dir.name / img_path.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(img_path, dst)
                
                # test 파일 복사
                for img_path in test_images:
                    dst = test_output_dir / class_dir.name / img_path.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(img_path, dst)

data_dir = '/home/hice1/kpk6/scratch/Datasets/caltech-101/101_ObjectCategories'  # 압축을 해제한 폴더 경로
output_dir = '/home/hice1/kpk6/scratch/Datasets/caltech-101/101_ObjectCategories_tmp'  # 준비된 데이터셋을 저장할 폴더 경로

prepare_caltech101_dataset(data_dir, output_dir)
