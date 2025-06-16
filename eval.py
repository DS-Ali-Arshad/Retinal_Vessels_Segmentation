import os
import sys
import argparse

import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import (
    accuracy_score, f1_score, jaccard_score,
    precision_score, recall_score
)

from metrics import dice_loss, dice_coef, iou, combined_loss

# ---- CONSTANTS ----
H, W = 512, 512
DEFAULT_THRESHOLD = 0.5


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_image(path):
    ori = cv2.imread(path, cv2.IMREAD_COLOR)
    img = ori.astype(np.float32) / 255.0
    return ori, img


def read_mask(path):
    ori = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    m = (ori.astype(np.float32) / 255.0).astype(np.int32)
    return ori, m


def load_data(path):
    x_paths = sorted(glob(os.path.join(path, "image", "*.png")))
    y_paths = sorted(glob(os.path.join(path, "mask",  "*.png")))
    if len(x_paths) != len(y_paths):
        raise RuntimeError("Number of images and masks do not match!")
    return x_paths, y_paths


def save_comparison(ori_x, ori_y, pred_mask, out_path):
    """ Stack: [RGB image | white bar | true mask | white bar | pred mask] """
    bar = np.ones((H, 10, 3), dtype=np.uint8) * 255

    # convert masks to 3-channel uint8
    true_rgb = np.stack([ori_y, ori_y, ori_y], axis=-1)
    pred_bin  = (pred_mask * 255).astype(np.uint8)
    pred_rgb  = np.stack([pred_bin, pred_bin, pred_bin], axis=-1)

    canvas = np.concatenate([ori_x, bar, true_rgb, bar, pred_rgb], axis=1)
    cv2.imwrite(out_path, canvas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference & metrics for DRIVE segmentation"
    )
    parser.add_argument("out_prefix",
        help="Base name for output folders (results & files)")
    parser.add_argument("--data", default=os.path.join("new_data","test"),
        help="Path to test dataset (contains image/ and mask/ subfolders)")
    parser.add_argument("--model", default=os.path.join("files","model.keras"),
        help="Path to the .keras model file")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
        help="Binarization threshold for model output")
    args = parser.parse_args()

    # prepare folders
    results_dir = f"{args.out_prefix}_results"
    files_dir   = f"{args.out_prefix}_files"
    create_dir(results_dir)
    create_dir(files_dir)

    # load model (inference only)
    with CustomObjectScope({
        'iou': iou,
        'dice_coef': dice_coef,
        'dice_loss': dice_loss,
        'combined_loss': combined_loss
    }):
        model = tf.keras.models.load_model(args.model, compile=False)

    # load test file paths
    test_x, test_y = load_data(args.data)

    scores = []
    for img_path, mask_path in tqdm(zip(test_x, test_y), total=len(test_x)):
        name = os.path.splitext(os.path.basename(img_path))[0]

        ori_x, x_in = read_image(img_path)
        ori_y, y_true = read_mask(mask_path)

        # model.predict gives shape (H,W,1) → squeeze to (H,W)
        pred_prob = model.predict(x_in[None, ...])[0, ..., 0]
        pred_mask = (pred_prob > args.threshold).astype(np.int32)

        # save side-by-side comparison
        out_png = os.path.join(results_dir, f"{name}.png")
        save_comparison(ori_x, ori_y, pred_mask, out_png)

        # flatten & compute metrics
        y_flat  = y_true.flatten()
        p_flat  = pred_mask.flatten()
        acc      = accuracy_score(y_flat, p_flat)
        f1       = f1_score(y_flat, p_flat)
        jaccard  = jaccard_score(y_flat, p_flat)
        recall   = recall_score(y_flat, p_flat)
        precision= precision_score(y_flat, p_flat)

        scores.append([name, acc, f1, jaccard, recall, precision])

    # aggregate
    arr = np.array([s[1:] for s in scores], dtype=np.float32)
    mean_vals = arr.mean(axis=0)
    print(f"Accuracy : {mean_vals[0]:.5f}")
    print(f"F1       : {mean_vals[1]:.5f}")
    print(f"Jaccard  : {mean_vals[2]:.5f}")
    print(f"Recall   : {mean_vals[3]:.5f}")
    print(f"Precision: {mean_vals[4]:.5f}")

    # save per-image scores
    df = pd.DataFrame(
        scores,
        columns=["Image", "Accuracy", "F1", "Jaccard", "Recall", "Precision"]
    )
    df.to_csv(os.path.join(files_dir, "score.csv"), index=False)
