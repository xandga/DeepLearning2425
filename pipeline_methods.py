# Standard libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import zipfile
import seaborn as sns
import tensorflow as tf

# Libraries for image processing
from glob import glob
from PIL import Image

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight

# Cnidaria MoblieNetV2 Train

#Function to augment the images
def augment_image(image, label):

    #Randomly change brightness
    image = tf.image.random_brightness(image, max_delta=0.2)

    #Apply geometric augmentations
    image = geometric_augmentation_layers(image, training=True) # Apply geometric augmentations
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, label

geometric_augmentation_layers = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),  # Randomly flip horizontally
    tf.keras.layers.RandomRotation(factor=0.12),  # Randomly rotate
    tf.keras.layers.RandomZoom(height_factor=(-0.35, 0.35), width_factor=(-0.35, 0.35)),  # Random zoom
    tf.keras.layers.RandomTranslation(height_factor=0.20, width_factor=0.20),  # Random shift
    tf.keras.layers.RandomContrast(factor=0.25)  # Contrast
], name="geometric_augmentations")




#Function to preprocess the images
def process_image(file_path, label, image_size=(224, 224)):
    image = tf.io.read_file(file_path) # Read the image file
    image = tf.image.decode_jpeg(image, channels=3) # Decode the JPEG image
    image = tf.image.resize(image, image_size) # Resize the image to the target size
    
    image = mobilenet_preprocess(image)  # Apply MobileNetV2 preprocessing
    return image, label

def calculate_minority_indices(train_df, label_to_index, threshold=24):
    """
    Calculate the indices of minority classes based on a threshold.

    Args:
        train_df (pd.DataFrame): The training DataFrame containing the 'family' column.
        label_to_index (dict): A mapping from family labels to integer indices.
        threshold (int): The maximum count for a class to be considered a minority.

    Returns:
        tf.Tensor: A TensorFlow constant containing the indices of minority classes.
    """
    family_counts = train_df['family'].value_counts()
    minority_families = family_counts[family_counts <= threshold].index.tolist()
    minority_indices = [label_to_index[fam] for fam in minority_families if fam in label_to_index]
    print("Minority class indices:", minority_indices)
    return tf.constant(minority_indices, dtype=tf.int32)

def oversample_minority_classes(train, minority_indices_tf, oversample_factor=2):
    """
    Oversample minority classes in the training dataset.

    Args:
        train (tf.data.Dataset): The training dataset containing file paths and labels.
        minority_indices_tf (tf.Tensor): A TensorFlow constant containing the indices of minority classes.
        oversample_factor (int): The factor by which to oversample the minority classes.

    Returns:
        tf.data.Dataset: A concatenated dataset with oversampled minority classes and majority classes.
    """
    # Filter the training dataset into minority and majority subsets
    minority_ds = train.filter(lambda fp, label: tf.reduce_any(tf.equal(label, minority_indices_tf)))
    majority_ds = train.filter(lambda fp, label: tf.logical_not(tf.reduce_any(tf.equal(label, minority_indices_tf))))

    # Repeat the minority dataset to oversample it
    minority_ds = minority_ds.repeat(oversample_factor)

    # Concatenate the majority and oversampled minority datasets
    # Note: It's advisable to re-shuffle after concatenation
    train = majority_ds.concatenate(minority_ds)
    train = train.shuffle(buffer_size=len(train), reshuffle_each_iteration=True, seed=42)

    return train

def create_class_weights(labels_train):
    y_train = np.array(labels_train)  
    class_weights = compute_class_weight(class_weight='balanced',classes=np.unique(y_train),y=y_train)

    class_weights_dict = dict(enumerate(class_weights))
    print("Computed class weights:", class_weights_dict)

    return class_weights_dict

def create_datasets(train_df, test_df):
    """
    Create file paths and map labels for training and test datasets.

    Args:
        train_df (pd.DataFrame): The training DataFrame containing 'file_path' and 'family' columns.
        test_df (pd.DataFrame): The test DataFrame containing 'file_path' and 'family' columns.
        root_dir (str): The root directory where the files are located.

    Returns:
        tuple: file_paths_train, labels_train, file_paths_test, labels_test, label_to_index
    """
    root_dir = "/root/DeepLearning2425/rare_species"

    # Append full file paths to training and test DataFrames
    train_df['full_path'] = train_df['file_path'].apply(lambda x: os.path.normpath(os.path.join(root_dir, x)))
    test_df['full_path'] = test_df['file_path'].apply(lambda x: os.path.normpath(os.path.join(root_dir, x)))

    file_paths_train = train_df['full_path'].tolist()
    labels_train = train_df['family'].tolist()

    file_paths_test = test_df['full_path'].tolist()
    labels_test = test_df['family'].tolist()

    # Map the labels to integers
    label_names = sorted(set(labels_train))  # Get the unique labels
    label_to_index = {name: i for i, name in enumerate(label_names)}  # Create a mapping from labels to integers
    labels_train = [label_to_index[label] for label in labels_train]
    labels_test = [label_to_index[label] for label in labels_test]

    train = tf.data.Dataset.from_tensor_slices((file_paths_train, labels_train))
    train = train.shuffle(buffer_size=len(file_paths_train), reshuffle_each_iteration=False, seed=42)

    test = tf.data.Dataset.from_tensor_slices((file_paths_test, labels_test))
    test = test.shuffle(buffer_size=len(file_paths_test), reshuffle_each_iteration=False, seed=42)

    return train, test, labels_train, labels_test, label_to_index

def preprocess_pipeline(train, test, batch_size=8):
    # --- Training Preprocess Pipeline ---
    train = train.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    train = train.cache()
    train = train.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    train = train.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # --- Test Preprocess Pipeline ---
    test = test.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    test = test.cache().batch(8).prefetch(tf.data.AUTOTUNE)

    return train, test

def minority_from_labels(labels, threshold=24):
    unique, counts = np.unique(labels, return_counts=True)
    return unique[counts <= threshold].tolist()

