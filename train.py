# Standard libraries
import os
import matplotlib.pyplot as plt
import zipfile
import seaborn as sns
import random
import sys
import json # for loading in config file

# Libraries for data manipulation and analysis
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# Libraries for image processing
from glob import glob
from PIL import Image

# Libraries for deep learning
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical, image_dataset_from_directory
from tensorflow.keras.regularizers import l2
# Functions for visualization
from visualization import plot_family_distribution_by_phylum, plot_cumulative_family_distribution


# if sys.args[1] == "wandb":
#     wandb.init(project="rare_species")

# Found in the sys module (import sys)
# sys.argv is a list of command-line arguments passed to a Python script.
# sys.argv[0] is the script name.
# sys.argv[1:] are the actual arguments passed.


# 1.) Check if we should use wandb or not
# for logging to wandb 
if sys.argv[1] == "wandb":
    import wandb # for logging to wandb 
    wandb.init(project="rare_species")
elif sys.argv[1] == "no_wandb":
    pass
else:
    print("Invalid argument. Use 'wandb' or 'no_wandb'")
    sys.exit(1)


# 2.) Check if we should use a parameter config file or not 
# defining default parameters
default_paramters = {
    'IMG_HEIGHT': 224,
    'IMG_WIDTH': 224,
    'BATCH_SIZE': 32
}

try:
    config_file = sys.argv[2]
    print(f"Using config file from input: {config_file}")
except IndexError:
    print("No config file provided using standard parameters")
    print(default_paramters)

# load in config_file
try:
    with open(config_file, 'r') as f:
        parametersconfig = json.load(f)
except FileNotFoundError:
    print(f"Config file {config_file} not found. Make sure the config file exists and is a valid json file")
    sys.exit(1)

# 3.) Check if we should use a picture path from sys.argv[3] or not
default_picture_path = 'chordata_images'
# Try to get picture path from sys.argv[3] if it exists
try:
    picture_path = sys.argv[3]
    print(f"Using picture path from input: {picture_path}")
except IndexError:
    print("No picture path provided. Using default.")
    picture_path = default_picture_path
    print(f"Using default picture path: {picture_path}")

# include a check if the picture path is a valid path
if not os.path.exists(picture_path):
    print(f"Picture path {picture_path} does not exist. Make sure the path exists")
    sys.exit(1)

if config_file is not None:
    if 'IMG_HEIGHT' in parametersconfig:
        IMG_HEIGHT = parametersconfig['IMG_HEIGHT']
        print(f"IMG_HEIGHT set to {IMG_HEIGHT} from config file")
    if 'IMG_WIDTH' in parametersconfig:
        IMG_WIDTH = parametersconfig['IMG_WIDTH']
        print(f"IMG_WIDTH set to {IMG_WIDTH} from config file")
    if 'BATCH_SIZE' in parametersconfig:
        BATCH_SIZE = parametersconfig['BATCH_SIZE']
        print(f"BATCH_SIZE set to {BATCH_SIZE} from config file")
else:
    print("No config file provided using default parameters")






# 4.) Load in the data with image_dataset_from_directory 
# 
train_ds = tf.keras.utils.image_dataset_from_directory(
    picture_path,                 # Path to the main folder
    labels='inferred',             # Inferred from subfolder names
    label_mode='categorical',      # We'll get one-hot encoded labels
    color_mode='rgb',              # We want 3 channels (RGB)
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=True,                  # Important for training
    validation_split=0.2,          # 20% validation
    subset='training',
    seed=123                       # For reproducibility of the split
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    picture_path,
    labels='inferred',
    label_mode='categorical',
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=True,
    validation_split=0.2,
    subset='validation',
    seed=123
)


# TODO: Implement wndb loggin to track the process on the website wandb.ai

# save the class_names
class_names = train_ds.class_names
print("Class names:", class_names)

# Apply the normalization layer to the datasets
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))


# creates new pictures by flipping, rotating, and zooming
# 1)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1), # 0.1 * 180 
    layers.RandomZoom(0.1), # 0.1 * 180 
])

# 2) 
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache()\
                   .shuffle(buffer_size=1000)\
                   .prefetch(buffer_size=AUTOTUNE)

val_ds   = val_ds.cache()\
                 .prefetch(buffer_size=AUTOTUNE)

# 3) Standardization (simple rescaling)
normalization_layer = layers.Rescaling(1./255)

# 4) Combine augmentation + standardization
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y))

# 5) Cache, shuffle, and prefetch
train_ds = train_ds.cache()\
                   .shuffle(1000)\
                   .prefetch(buffer_size=AUTOTUNE)

val_ds   = val_ds.cache()\
                 .prefetch(buffer_size=AUTOTUNE)

# 6) Example model
model = tf.keras.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(166, activation='softmax')  # 166 classes
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 7) Train
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10, 
    callbacks=[
                      WandbMetricsLogger(log_freq=5),
                      WandbModelCheckpoint("models")
                    ]
    

)
