import os
from PIL import Image, UnidentifiedImageError

def clean_corrupted_images(directory: str):
    """
    Recursively walks through a directory and deletes corrupted .jpg images.
    
    Args:
    directory (str): The directory to walk through.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.jpg'):
                file_path = os.path.join(root, file)
                try:
                    img = Image.open(file_path)
                    img = img.convert('RGB')
                    img.verify()  # Verify that it is, in fact, an image
                except:
                    print(f"Deleting corrupted image: {file_path}")
                    os.remove(file_path)

directory = '/data/sophia/caltech-101/101_ObjectCategories_tmp'
clean_corrupted_images(directory)
