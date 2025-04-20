#!/usr/bin/env python3
"""
train_yolov8_chordata_fp32.py
Fine‑tune YOLOv8‑s classifier on 166 animal families, 40 epochs,
with live wandb tracking.  No layer‑freezing; AMP disabled (MPS bug‑free).
"""

import argparse, random, shutil
from pathlib import Path
import torch, wandb
from ultralytics import YOLO


# ---------- helpers -------------------------------------------------- #
def split_train_val(root: Path, pct=0.8, seed=42):
    (root/"train").mkdir(parents=True, exist_ok=True)
    (root/"val").mkdir(parents=True, exist_ok=True)
    if any((root/"train").iterdir()) and any((root/"val").iterdir()):
        return
    random.seed(seed)
    for cls in [d for d in root.iterdir() if d.is_dir() and d.name not in {"train","val"}]:
        imgs = sorted(cls.glob("*")); random.shuffle(imgs)
        n = int(len(imgs)*pct)
        for split, files in (("train",imgs[:n]), ("val",imgs[n:])):
            dst = root/split/cls.name; dst.mkdir(parents=True, exist_ok=True)
            for f in files: shutil.copy2(f, dst/f.name)

def sanity(root: Path):
    assert (root/"train").is_dir() and (root/"val").is_dir(), \
        "Expect pre‑split train/ and val/ folders."


# ---------- main ----------------------------------------------------- #
def main(a):
    root = Path(a.data_root).expanduser()
    split_train_val(root); sanity(root)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    run = wandb.init(project=a.wandb_project, name=a.wandb_run, config=vars(a))

    model = YOLO("yolov8s-cls.pt")

    model.train(                      # single 40‑epoch run, FP32
        data=str(root),
        imgsz=a.img,
        batch=a.batch,
        epochs=a.epochs,
        device=dev,
        optimizer="AdamW",
        lr0=a.lr,
        patience=a.patience,
        amp=False,                    # <-- critical change
        project=a.wandb_project,
        name=a.wandb_run,
        exist_ok=True,
        workers=a.workers,
    )

    metrics = model.val(data=str(root), split="val", plots=True)
    run.log({f"val/{k}": v for k,v in metrics.results_dict.items()})
    run.finish()
    print("✅ training done; see wandb for metrics.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--img", type=int, default=224)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--wandb_project", default="yolov8-chordata")
    p.add_argument("--wandb_run", default="yolov8s-40ep-fp32")
    main(p.parse_args())
