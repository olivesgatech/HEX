import os
from torchvision.datasets.utils import download_url, extract_archive
import shutil

def download_and_extract_oxford_pets(data_dir):
    """Downloads and extracts the Oxford-IIIT Pets dataset."""
    url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    annotations_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

    os.makedirs(data_dir, exist_ok=True)

    # Download and extract images
    tar_path = os.path.join(data_dir, "images.tar.gz")
    if not os.path.exists(tar_path):
        print(f"Downloading {url}...")
        download_url(url, data_dir, "images.tar.gz")
    else:
        print(f"{tar_path} already exists. Skipping download.")

    if not os.path.exists(os.path.join(data_dir, "images")):
        print(f"Extracting {tar_path}...")
        extract_archive(tar_path, data_dir)
    else:
        print(f"{os.path.join(data_dir, 'images')} already exists. Skipping extraction.")

    # Download and extract annotations
    annos_path = os.path.join(data_dir, "annotations.tar.gz")
    if not os.path.exists(annos_path):
        print(f"Downloading {annotations_url}...")
        download_url(annotations_url, data_dir, "annotations.tar.gz")
    else:
        print(f"{annos_path} already exists. Skipping download.")

    if not os.path.exists(os.path.join(data_dir, "annotations")):
        print(f"Extracting {annos_path}...")
        extract_archive(annos_path, data_dir)
    else:
        print(f"{os.path.join(data_dir, 'annotations')} already exists. Skipping extraction.")

    print("Download and extraction completed.")

def split_dataset(data_dir):
    """Splits the dataset into train and test sets."""
    annotations_dir = os.path.join(data_dir, 'annotations')
    train_file = os.path.join(annotations_dir, 'trainval.txt')
    test_file = os.path.join(annotations_dir, 'test.txt')

    with open(train_file, 'r') as f:
        train_list = [line.split()[0] for line in f.readlines()]

    with open(test_file, 'r') as f:
        test_list = [line.split()[0] for line in f.readlines()]

    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for image_name in train_list:
        src = os.path.join(data_dir, 'images', image_name + '.jpg')
        dst_dir = os.path.join(train_dir, image_name.split('_')[0])
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(src))
        shutil.copyfile(src, dst)

    for image_name in test_list:
        src = os.path.join(data_dir, 'images', image_name + '.jpg')
        dst_dir = os.path.join(test_dir, image_name.split('_')[0])
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(src))
        shutil.copyfile(src, dst)

    print("Dataset split into train and test sets.")

download_and_extract_oxford_pets("/storage/ice1/7/2/skim3513/oxfordpets")
split_dataset("/storage/ice1/7/2/skim3513/oxfordpets")