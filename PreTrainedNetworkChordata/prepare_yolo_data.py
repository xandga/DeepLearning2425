#!/usr/bin/env python3
import os
import sys
import shutil
import random
import yaml

"""
Quick explanation of what this script does:

1) It takes a directory containing subdirectories of images (each subdirectory representing a class).
2) It creates a train/ and val/ directory structure, where each class has its own subdirectory.
3) It splits the images into training and validation sets based on a specified fraction (default is 0.8).
4) It copies the images into the appropriate train/ and val/ directories.
5) It generates a data.yaml file that contains the paths to the train and val directories, the number of classes, and the class names.
6) It prints out the number of training and validation images for each class.

"""
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare(cleaned_dir, train_frac=0.8):
    cleaned_dir = Path(cleaned_dir)
    assert cleaned_dir.is_dir(), f"{cleaned_dir} is not a directory"

    # 1) Find all class subfolders
    class_dirs = [d for d in cleaned_dir.iterdir() if d.is_dir()]
    class_dirs.sort()
    class_names = [d.name for d in class_dirs]

    # 2) Create train/ and val/ directories
    train_root = cleaned_dir / "train"
    val_root   = cleaned_dir / "val"
    for root in (train_root, val_root):
        root.mkdir(exist_ok=True)
        for cls in class_names:
            (root / cls).mkdir(exist_ok=True)

    # 3) Split & copy images
    for cls in class_names:
        src_dir = cleaned_dir / cls
        images = [f for f in src_dir.iterdir() if f.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.gif')]
        if not images:
            continue

        train_imgs, val_imgs = train_test_split(
            images,
            train_size=train_frac,
            shuffle=True,
            random_state=42
        )

        for img in train_imgs:
            shutil.copy(img, train_root / cls / img.name)
        for img in val_imgs:
            shutil.copy(img, val_root   / cls / img.name)

        print(f"Class '{cls}': {len(train_imgs)} train, {len(val_imgs)} val")

    # 4) Write data.yaml
    data = {
        'train': str(train_root.resolve()),
        'val':   str(val_root.resolve()),
        'nc':    len(class_names),
        'names': class_names
    }
    with open(cleaned_dir / 'data.yaml', 'w') as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"\nCreated data.yaml with {len(class_names)} classes at {cleaned_dir/'data.yaml'}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python prepare_yolo_data.py <cleaned_dataset_dir> [train_fraction]")
        sys.exit(1)
    cleaned_dir = sys.argv[1]
    train_frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    prepare(cleaned_dir, train_frac)

