import cv2
import os
import pandas as pd
from pathlib import path
from typing import Dict, List
import os


def build_file_index(data_dir: str) -> Dict[str, List[str]]:
    """
    Scans `data_dir` for subfolders and returns a dict mapping
    each folder name to a list of relative paths "folder/filename.ext".
    """
    base = Path(data_dir)
    index: Dict[str, List[str]] = {}

    for class_folder in base.iterdir():
        if not class_folder.is_dir():
            continue

        # Gather all files in this class‑folder
        files = [
            f"{class_folder.name}/{p.name}"
            for p in class_folder.iterdir()
            if p.is_file()
        ]
        index[class_folder.name] = files

    return index

def process_image_with_clahe(image_path, size=(224, 224)):
    """
    Loads an image, resizes it, and applies CLAHE.
    Returns the resized image and the CLAHE-enhanced version.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load: {image_path}")
            return None, None

        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize
        resized_img = cv2.resize(img_rgb, size, interpolation=cv2.INTER_AREA)

        # Convert to LAB and apply CLAHE
        lab = cv2.cvtColor(resized_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

        return resized_img, img_clahe

    except Exception as e:
        print(f" Error processing {image_path}: {e}")
        return None, None   

path_dict = build_file_index('../cleaned_chordata_images')

for phylum_family_class, path_list in path_dict.items():

    # create a dictionary that saves the newly images in a folder
    directory = 'cahe_images'
    os.path.join(directory, key)



    # remove the classname/ from the path file 
    class_folder_str = phylum_family_class + '/'
    # 
    file_names = [path.replace(class_folder_str, '') for path in path_list]


    # apply the cahe algorithm 

    map(cahe, a)















chordata_train[["resized_image", "image_clahe"]] = chordata_train["file_path"].progress_apply(
    lambda path: pd.Series(process_image_with_clahe(os.path.join(image_folder, path)))
)