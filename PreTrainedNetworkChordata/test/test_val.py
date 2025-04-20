import pandas as pd
import ultralytics
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# read in the yolo model that I have trained
# 1. Define paths (Adjust if necessary based on the script's location relative to these paths)
model_path = '../yolov8-chordata/yolov8s-40ep/weights/best.pt'
data_root = 'additional_data_val' # Directory containing train/ and val/ subdirs

# 2. Load the trained YOLO model
print(f"Loading model from: {model_path}")
model = YOLO(model_path)

# 3. Run validation on the 'val' split
# plots=True generates confusion matrix, etc.
print(f"Running validation on data from: {data_root}")
metrics = model.val(data=data_root, split='val', plots=True)

# print(f"Running test on data from: {data_root}")
# metrics = model.val(data=data_root, split= 'train', plots = True)

# 4. Print key metrics from the results dictionary
print("\n--- Validation Metrics ---")
try:
    top1_acc = metrics.results_dict['metrics/accuracy_top1']
    top5_acc = metrics.results_dict['metrics/accuracy_top5']
    print(f"  Top-1 Accuracy: {top1_acc:.4f}")
    print(f"  Top-5 Accuracy: {top5_acc:.4f}")
except KeyError as e:
    print(f"  Could not find expected accuracy metric: {e}")
    print(f"  Available metrics: {metrics.results_dict.keys()}")

# Calculate F1 Scores from Confusion Matrix
try:
    conf_matrix = metrics.confusion_matrix.matrix
    num_classes = conf_matrix.shape[0]
    tp = np.diag(conf_matrix)
    fp = conf_matrix.sum(axis=0) - tp
    fn = conf_matrix.sum(axis=1) - tp
    tn = conf_matrix.sum() - (tp + fp + fn) # Not needed for F1, but good to have

    precision = tp / (tp + fp + 1e-9) # Add epsilon for stability
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

    macro_f1 = np.mean(f1)
    
    support = conf_matrix.sum(axis=1) # Number of true instances per class
    weighted_f1 = np.average(f1, weights=support)

    print(f"  Macro F1-Score: {macro_f1:.4f}")
    print(f"  Weighted F1-Score: {weighted_f1:.4f}")

except AttributeError:
     print("  Could not find confusion matrix in metrics object.")
except Exception as e:
    print(f"  Error calculating F1 scores: {e}")

# 5. Inform the user where plots are saved
# The directory usually looks like runs/classify/val*
print(f"\nPlots (including confusion matrix) saved to: {metrics.save_dir}")
print("--- Evaluation Complete ---")
