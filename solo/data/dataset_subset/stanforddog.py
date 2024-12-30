import os
from scipy.io import loadmat
from shutil import copyfile
from torchvision.datasets.utils import download_url, extract_archive

def download_and_extract_stanford_dogs(data_dir):
    """Downloads and extracts the Stanford Dogs dataset."""
    url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
    annos_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar"

    os.makedirs(data_dir, exist_ok=True)

    # Download and extract images
    tar_path = os.path.join(data_dir, "images.tar")
    if not os.path.exists(tar_path):
        print(f"Downloading {url}...")
        download_url(url, data_dir, "images.tar")
    else:
        print(f"{tar_path} already exists. Skipping download.")

    image_dir = os.path.join(data_dir, "images")
    if not os.path.exists(image_dir):
        print(f"Extracting {tar_path}...")
        extract_archive(tar_path, data_dir)
    else:
        print(f"{image_dir} already exists. Skipping extraction.")

    # Download and extract annotations
    annos_path = os.path.join(data_dir, "lists.tar")
    if not os.path.exists(annos_path):
        print(f"Downloading {annos_url}...")
        download_url(annos_url, data_dir, "lists.tar")
    else:
        print(f"{annos_path} already exists. Skipping download.")

    if not os.path.exists(os.path.join(data_dir, "lists")):
        print(f"Extracting {annos_path}...")
        extract_archive(annos_path, data_dir)
    else:
        print(f"{os.path.join(data_dir, 'lists')} already exists. Skipping extraction.")

    print("Download and extraction completed.")

def split_dataset(data_dir):
    """Splits the dataset into train and test sets."""
    train_list = loadmat(os.path.join(data_dir, 'train_list.mat'))['file_list']
    test_list = loadmat(os.path.join(data_dir, 'test_list.mat'))['file_list']

    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for item in train_list:
        src = os.path.join(data_dir, 'Images', item[0][0])
        dst_dir = os.path.join(train_dir, os.path.basename(os.path.dirname(src)))
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(src))
        copyfile(src, dst)

    for item in test_list:
        src = os.path.join(data_dir, 'Images', item[0][0])
        dst_dir = os.path.join(test_dir, os.path.basename(os.path.dirname(src)))
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(src))
        copyfile(src, dst)

    print("Dataset split into train and test sets.")

# Example usage
data_dir = "/home/hice1/skim3513/scratch/stanford_dogs"
download_and_extract_stanford_dogs(data_dir)
split_dataset(data_dir)
