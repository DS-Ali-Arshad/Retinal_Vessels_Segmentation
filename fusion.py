# fusion.py
# Script to load trained thin and thick vessel segmentation models,
# predict on input images, fuse their outputs, and save combined masks.

import os
import sys
import argparse
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope

# If your models used custom losses/metrics, import them here:
from metrics import dice_loss, dice_coef, iou, combined_loss, soft_cldice_loss

#----------------------------------------------------------------
# Constants
H, W = 512, 512  # input size for the models
DEFAULT_THRESHOLD = 0.5

#----------------------------------------------------------------
# Utility functions

def create_dir(path):
    os.makedirs(path, exist_ok=True)


def read_image(path):
    """
    Reads an image from disk, resizes to (H,W), returns BGR uint8
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.resize(img, (W, H))
    return img


def preprocess(img):
    """
    Normalize image to float32 in [0,1]
    """
    x = img.astype(np.float32) / 255.0
    return x


def binarize_mask(prob_map, threshold=DEFAULT_THRESHOLD):
    """
    Converts probability map to binary mask (0 or 255)
    """
    mask = (prob_map > threshold).astype(np.uint8) * 255
    return mask

#----------------------------------------------------------------
# Main fusion logic

def main(args):
    # Prepare output directory
    create_dir(args.output_dir)

    # Load thin and thick models
    # If models include custom objects, load within CustomObjectScope
    with CustomObjectScope({
        'dice_loss': dice_loss,
        'dice_coef': dice_coef,
        'iou': iou,
        'combined_loss': combined_loss,
        'soft_cldice_loss': soft_cldice_loss()
    }):
        thin_model = tf.keras.models.load_model(args.thin_model, compile=False)
        thick_model = tf.keras.models.load_model(args.thick_model, compile=False)

    # Gather input images
    img_paths = sorted(glob(os.path.join(args.data, 'image', '*.png')))
    if not img_paths:
        print(f"No images found in {args.data}/image")
        sys.exit(1)

    # Process each image
    for img_path in img_paths:
        name = os.path.splitext(os.path.basename(img_path))[0]
        # Read & preprocess
        ori = read_image(img_path)
        x_in = preprocess(ori)

        # Predict thin vessels
        thin_prob = thin_model.predict(x_in[None, ...])[0, ..., 0]
        thin_mask = binarize_mask(thin_prob, args.threshold)

        # Predict thick vessels
        thick_prob = thick_model.predict(x_in[None, ...])[0, ..., 0]
        thick_mask = binarize_mask(thick_prob, args.threshold)

        # Fuse by logical OR
        combined = ((thin_mask > 0) | (thick_mask > 0)).astype(np.uint8) * 255

        # Save fused mask
        out_path = os.path.join(args.output_dir, f"{name}_fused.png")
        cv2.imwrite(out_path, combined)
        print(f"Saved fused mask: {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fuse thin and thick vessel segmentation outputs'
    )
    parser.add_argument(
        'data', help='Path to dataset folder containing image/ subfolder'
    )
    parser.add_argument(
        '--thin-model', required=True,
        help='Path to the trained thin vessel model (.keras)'
    )
    parser.add_argument(
        '--thick-model', required=True,
        help='Path to the trained thick vessel model (.keras)'
    )
    parser.add_argument(
        '--threshold', type=float, default=DEFAULT_THRESHOLD,
        help='Binarization threshold for predictions'
    )
    parser.add_argument(
        '--output-dir', default='fused_results',
        help='Directory to save fused masks'
    )
    args = parser.parse_args()

    main(args)
