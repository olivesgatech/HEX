import os
import shutil

dataset_root = '/home/hice1/kpk6/scratch/Datasets/CUB_200_2011'
images_txt_path = os.path.join(dataset_root, 'images.txt')
train_test_split_path = os.path.join(dataset_root, 'train_test_split.txt')

train_folder = os.path.join(dataset_root, 'train')
test_folder = os.path.join(dataset_root, 'test')

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

with open(images_txt_path, 'r') as f:
    images = f.readlines()

with open(train_test_split_path, 'r') as f:
    splits = f.readlines()

for image, split in zip(images, splits):
    image_id, image_path = image.strip().split()
    split_id, is_train = split.strip().split()
    
    
    src_path = os.path.join(dataset_root, 'images', image_path)
    
    if int(is_train):
        dst_path = os.path.join(train_folder, image_path)
    else:
        dst_path = os.path.join(test_folder, image_path)
    
    
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    
    
    shutil.copyfile(src_path, dst_path)

print('Train/test split completed.')
