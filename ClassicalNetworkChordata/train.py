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

from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import wandb

 
# 2.) Check if we should use a parameter config file or not 
# defining default parameters
default_paramters = {
    "data_path": "cleaned_dataset",
    "batch_size": 32,
    "img_height": 224,
    "img_width": 224,
    "epochs": 10,
    "learning_rate": 0.003,
    "num_classes": 166
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

# check if parametersconfig available if not use default config
if parametersconfig is None:
    parametersconfig = default_paramters

# 1.) Check if we should use wandb or not
# for logging to wandb 
if sys.argv[1] == "wandb":
    import wandb # for logging to wandb 
    wandb.init(
        project=parametersconfig.get("project", "default_project"), # Project name if provided else default_project
        config=parametersconfig
    )


# Start a run, tracking hyperparameters

# check for wandb object
config = wandb.config  # for convenience

# 4.) Load in the data with image_dataset_from_directory 
# 
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=config["data_path"],                 # Path to the main folder
    labels='inferred',             # Inferred from subfolder names
    label_mode='categorical',      # We'll get one-hot encoded labels
    color_mode='rgb',              # We want 3 channels (RGB)
    batch_size=config["batch_size"],
    image_size=(config["img_height"], config["img_width"]),
    validation_split=0.2,          # 20% validation
    subset='training',
    seed=123,                    # For reproducibility of the split
    shuffle=True         # Important for training
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    directory=config["data_path"],                 # Path to the main folder
    labels='inferred',             # Inferred from subfolder names
    label_mode='categorical',      # We'll get one-hot encoded labels
    color_mode='rgb',              # We want 3 channels (RGB)
    batch_size=config["batch_size"],
    image_size=(config["img_height"], config["img_width"]),
    validation_split=0.2,      
    subset='validation',
    seed=123,
    shuffle=True            # Important for training
)

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

# 6) Example model
model = tf.keras.Sequential([
    layers.Input(shape=(config['img_height'], config['img_height'], 3)),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(config['num_classes'], activation='softmax')  # 166 classes

])

# # 3) Build a quick model
# model = tf.keras.Sequential([
#     layers.Input(shape=(config["img_height"], config["img_width"], 3)),
#            # Input shape: e.g., (224, 224, 3) or any
#         # 1) A single small conv layer with 8 filters
#         layers.Conv2D(8, kernel_size=3, activation='relu'),
#         layers.MaxPooling2D(),
#         # 2) Flatten
#         layers.Flatten(),
#         # 3) A small dense layer
#         layers.Dense(32, activation='relu'),
#         # 4) Output layer for classification
#         layers.Dense(config['num_classes'], activation='softmax')
# ])

# 4) Compile the model
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=config.learning_rate)
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 5) Train the model, logging to wandb
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=config.epochs,
    callbacks=[
        WandbMetricsLogger(log_freq=5),     # log metrics every 5 batches
        WandbModelCheckpoint("models")      # save model checkpoints
    ]
)

# 6) Optional: Save final model or log artifacts
model.save("final_model")
wandb.finish()





# # 3) Standardization (simple rescaling)
# normalization_layer = layers.Rescaling(1./255)

# # 4) Combine augmentation + standardization
# train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
# train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y))

# # 5) Cache, shuffle, and prefetch
# train_ds = train_ds.cache()\
#                    .shuffle(1000)\
#                    .prefetch(buffer_size=AUTOTUNE)

# val_ds   = val_ds.cache()\
#                  .prefetch(buffer_size=AUTOTUNE)

# 6) Example model


# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # 7) Train
# model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=10, 
#     callbacks=[
#                       WandbMetricsLogger(log_freq=5),
#                       WandbModelCheckpoint("models")
#                     ]
    

# )


