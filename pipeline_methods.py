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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

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

def minority_from_labels(labels, threshold=24):
    unique, counts = np.unique(labels, return_counts=True)
    return unique[counts <= threshold].tolist()


# Cnidaria preprocessing
def process_image(image, label):
    image = mobilenet_preprocess(image)  # Apply MobileNetV2 preprocessing
    return image, label

def cnidaria_preprocess(test, batch_size=8):

    label_to_index = {
        label: idx 
        for idx, label in enumerate(sorted(test["family"].unique()))
    }

    X_test  = np.stack(test["clahe_image"].values).astype("float32")
    y_test  = np.array([label_to_index[label] for label in test["family"]])
    
    test = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


    return test

# Arthtropoda preprocessing
import os
import numpy as np
import tensorflow as tf

def arthropoda_preprocess(df, image_root_dir, batch_size=8):
    """
    Build a tf.data.Dataset for arthropoda test data.

    Args:
      df: pd.DataFrame with columns ["family","file_path"]
      image_root_dir: str, base directory to prepend to file_path
      batch_size: int

    Returns:
      A tf.data.Dataset yielding (image_tensor, label) batches.
    """
    # 1) Build a label→index map
    label_to_index = {
        label: idx 
        for idx, label in enumerate(sorted(df["family"].unique()))
    }

    # 2) Full file paths and numeric labels arrays
    file_paths = df["file_path"]\
        .apply(lambda p: os.path.join(image_root_dir, p))\
        .values.tolist()
    y_labels   = np.array([label_to_index[f] for f in df["family"]], dtype=np.int32)

    # 3) Create a Dataset of (path, label)
    ds = tf.data.Dataset.from_tensor_slices((file_paths, y_labels))

    # 4) Load, cast to float32, then preprocess
    def _load_and_process(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))            # ensure correct size
        img = tf.cast(img, tf.float32)                    # ← important!
        img = mobilenet_preprocess(img)                   # now safe to divide by floats
        return img, label

    ds = (
        ds
        .map(_load_and_process, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds




# Mollusca preprocessing 
def create_generators(test_df, image_root_dir, 
                      image_size=(128, 128), batch_size=16, 
                      x_col='filepath', y_col='family'):
    """
    Creates train, validation, and test data generators.

    Args:
        train_df (pd.DataFrame): DataFrame for training and validation.
        test_df (pd.DataFrame): DataFrame for testing.
        image_root_dir (str): Path to the directory where images are stored.
        image_size (tuple): Target size for images.
        batch_size (int): Batch size for generators.
        x_col (str): Column name for image filepaths.
        y_col (str): Column name for labels.

    Returns:
        train_generator, val_generator, test_generator
    """
    # Prepend full path to 'file_path' column if needed
    # train_df = train_df.copy()
    test_df = test_df.copy()
    
    # train_df[x_col] = train_df['file_path'].apply(lambda x: os.path.join(image_root_dir, x))
    test_df[x_col] = test_df['file_path'].apply(lambda x: os.path.join(image_root_dir, x))

    # # Data generators
    # train_datagen = ImageDataGenerator(
    #     rescale=1.0 / 255,
    #     rotation_range=15,
    #     zoom_range=0.1,
    #     horizontal_flip=True,
    #     validation_split=0.2
    # )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # # Train generator
    # train_generator = train_datagen.flow_from_dataframe(
    #     dataframe=train_df,
    #     x_col=x_col,
    #     y_col=y_col,
    #     target_size=image_size,
    #     class_mode='categorical',
    #     batch_size=batch_size,
    #     subset='training',
    #     shuffle=True,
    #     seed=4
    # )

    # # Validation generator
    # val_generator = train_datagen.flow_from_dataframe(
    #     dataframe=train_df,
    #     x_col=x_col,
    #     y_col=y_col,
    #     target_size=image_size,
    #     class_mode='categorical',
    #     batch_size=batch_size,
    #     subset='validation',
    #     shuffle=True,
    #     seed=4
    # )

    # Test generator
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col=x_col,
        y_col=y_col,
        target_size=image_size,
        class_mode='categorical',
        batch_size=1,
        shuffle=False
    )

    return test_generator

# Mollusca side by side preprocessing visualization
def visualize_pipeline_processed(train_df, image_root_dir, image_size=(128, 128), batch_size=1, num_samples=5, x_col='filepath'):
    """
    Shows side-by-side comparison of original vs resized+augmented images using the same train_datagen as in training.

    Args:
        train_df (pd.DataFrame): DataFrame for training images.
        image_root_dir (str): Root directory where images are stored.
        image_size (tuple): Target size for resizing (default (128, 128)).
        batch_size (int): Batch size for preview generator (default 1).
        num_samples (int): Number of samples to visualize (default 5).
        x_col (str): Column containing the full image paths (default 'filepath').
    """

    # Copy to avoid modifying original DataFrame
    sample_df = train_df.sample(n=num_samples, random_state=4).reset_index(drop=True)
    sample_df[x_col] = sample_df['file_path'].apply(lambda x: os.path.join(image_root_dir, x))

    # --- Define train_datagen exactly like your real augmentation during training ---
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2  # even if validation split is not used here, it's fine to match
    )

    # Generator for preview
    preview_generator = train_datagen.flow_from_dataframe(
        dataframe=sample_df,
        x_col=x_col,
        y_col=None,
        class_mode=None,
        target_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )

    plt.figure(figsize=(10, num_samples * 2.5))

    for i in range(num_samples):
        # Original image (raw, not resized or augmented)
        original_path = sample_df.loc[i, x_col]
        original = load_img(original_path)

        # Processed image (resized + augmented + normalized)
        processed = next(preview_generator)[0]
        processed = np.clip(processed, 0, 1)  # Keep pixel values in valid range [0,1]

        # Plot original
        plt.subplot(num_samples, 2, 2 * i + 1)
        plt.imshow(original)
        plt.title("Original")
        plt.axis('off')

        # Plot processed
        plt.subplot(num_samples, 2, 2 * i + 2)
        plt.imshow(processed)
        plt.title(f"Processed (Augmented {image_size[0]}x{image_size[1]})")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_loss(history):
    """
    Plot the training and validation loss over epochs.

    Parameters:
    - history: History object returned by model.fit() containing training/validation loss values.

    Returns:
    - None (displays a plot of training and validation loss)
    """
    plt.figure(figsize=(7, 2))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

# f1 score evaluation
def evaluate_f1(model, test_dataset):
    """
    Computes the F1 score for a Keras model given a test test_dataset.

    Parameters:
    - model: Trained model.
    - test_dataset: Test test_dataset.

    Returns:
    - f1: Macro-averaged F1 score.
    """
    y_pred_probs = model.predict(test_dataset, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_dataset.classes
    f1 = f1_score(y_true, y_pred, average='macro')
    
    return f1

def show_results_phylum(test_ds, results, label_to_index):
    """
    test_ds:        tf.data.Dataset yielding (x_batch, y_batch)
    results:        numpy array, shape (N, C)
    label_to_index: dict mapping family_name -> index (0..C-1)
    """
    # --- 1) Predictions & ground truth ---
    y_pred = np.argmax(results, axis=1)
    y_true = np.concatenate([y_batch.numpy() for _, y_batch in test_ds], axis=0)

    # --- 2) Build an ordered list of family names, length = number of output classes ---
    index_to_label = {idx: fam for fam, idx in label_to_index.items()}
    C = results.shape[1]
    families = [index_to_label.get(i, f"Class_{i}") for i in range(C)]

    # --- 3) Compute & print metrics ---
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='macro')

    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro F1 : {f1:.4f}\n")
    print("Classification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=families,
        digits=4
    ))

    # --- 4) Confusion matrix & plot ---
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', aspect='auto')
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Family")
    ax.set_ylabel("True Family")

    ax.set_xticks(np.arange(C))
    ax.set_yticks(np.arange(C))
    ax.set_xticklabels(families, rotation=45, ha="right")
    ax.set_yticklabels(families)

    thresh = cm.max() / 2
    for i in range(C):
        for j in range(C):
            ax.text(
                j, i, cm[i, j],
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

    return acc, f1