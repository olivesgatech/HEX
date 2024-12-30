# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import inspect
import logging
import os
import argparse
import shutil

def normalize_tin_val_folder_structure(path,
                                       images_folder='images',
                                       annotations_file='val_annotations.txt'):
    # Check if files/annotations are still there to see
    # if we already run reorganize the folder structure.
    images_folder = os.path.join(path, images_folder)
    annotations_file = os.path.join(path, annotations_file)

    # Exists
    if not os.path.exists(images_folder) \
       and not os.path.exists(annotations_file):
        if not os.listdir(path):
            raise RuntimeError('Validation folder is empty.')
        return

    # Parse the annotations
    with open(annotations_file) as f:
        for line in f:
            values = line.split()
            img = values[0]
            label = values[1]
            img_file = os.path.join(images_folder, values[0])
            label_folder = os.path.join(path, label)
            os.makedirs(label_folder, exist_ok=True)
            try:
                shutil.move(img_file, os.path.join(label_folder, img))
            except FileNotFoundError:
                continue

    os.sync()
    assert not os.listdir(images_folder)
    shutil.rmtree(images_folder)
    os.remove(annotations_file)
    os.sync()

def parse_option():
    parser = argparse.ArgumentParser('argument for generating ImageNet-100')

    parser.add_argument('--source_folder', type=str,
     default="/storage/ice1/shared/d-pace_community/makerspace-datasets/IMAGE/Imagenet2012/val/", help='folder of ImageNet-1K dataset')
    parser.add_argument('--target_folder', type=str,
     default='/home/hice1/kpk6/scratch/Datasets/ImageNet100/val/', help='folder of ImageNet-100 dataset')
    parser.add_argument('--target_class', type=str,
     default='/home/hice1/kpk6/solo-learn/solo/data/dataset_subset/imagenet100_classes.txt', help='class file of ImageNet-100')

    opt = parser.parse_args()

    return opt

def generate_data(source_folder, target_folder, target_class):
    f = []
    txt_data = open(target_class, "r") 
    for ids, txt in enumerate(txt_data):
        s = str(txt.split('\n')[0])
        f.append(s)
    

    for ids, dirs in enumerate(os.listdir(source_folder)):
        for tg_class in f:
            if dirs == tg_class:
                print('{} is transferred'.format(dirs))
                shutil.copytree(os.path.join(source_folder,dirs), os.path.join(target_folder,dirs)) 


if __name__ == "__main__":
    #opt = parse_option()
    #generate_data(opt.source_folder, opt.target_folder, opt.target_class)
    normalize_tin_val_folder_structure(path='/home/hice1/kpk6/scratch/Datasets/tiny-imagenet-200/val/',images_folder='images',annotations_file = 'val_annotations.txt')
