import os
import sys
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize, remove_small_objects, binary_dilation, disk, label
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops
from skimage.io import imsave

def mask_to_thin_thick(img, diameter_threshold=None, min_length=None, eccentricity_threshold=0.99, size_threshold_factor=4):
    """
    Separate an image mask into thin and thick vessels with adaptive filtering.
    
    Parameters:
    - img: Input binary mask
    - diameter_threshold: Threshold for vessel diameter (auto-calculated if None)
    - min_length: Minimum length for thick vessels (auto-calculated if None)
    - eccentricity_threshold: Threshold for shape elongation (default 0.8)
    - size_threshold_factor: Factor to determine adaptive size threshold (default 5)
    
    Returns:
    - thin_mask: Binary mask of thin vessels
    - thick_mask: Binary mask of thick vessels
    - (and computed thresholds for reference)
    """
    img_bin = img > 0
    skeleton = skeletonize(img_bin)
    edt = distance_transform_edt(img_bin)
    diameter_map = edt * 2 * skeleton

    # Automatic thresholds if not provided
    diam_values = diameter_map[diameter_map > 0]
    if diameter_threshold is None:
        if len(diam_values) == 0:
            diameter_threshold = 3  # Fallback value
        else:
            diameter_threshold = np.percentile(diam_values, 40)  # 40th percentile for balance
    total_pixels = np.sum(img_bin)
    if min_length is None:
        min_length = max(10, int(total_pixels * 0.002))

    # Vessel splitting
    thin_skel = (diameter_map > 0) & (diameter_map <= diameter_threshold)
    thin_mask = binary_dilation(thin_skel, disk(int(np.ceil(diameter_threshold/2))))
    thin_mask = thin_mask & img_bin

    # Label connected components
    labeled_thin = label(thin_mask)

    # Adaptive size threshold for filtering noise
    size_threshold = max(5, min_length // size_threshold_factor)

    # Filter: Remove only small AND circular components
    for region in regionprops(labeled_thin):
        if region.area < size_threshold and region.eccentricity < eccentricity_threshold:
            labeled_thin[labeled_thin == region.label] = 0

    thin_mask = labeled_thin > 0
    thin_mask = remove_small_objects(thin_mask, min_size=1)  # Clean up single pixels

    thick_mask = img_bin & (~thin_mask)
    thick_mask = remove_small_objects(thick_mask, min_size=min_length)

    return thin_mask, thick_mask, diameter_threshold, min_length, eccentricity_threshold, size_threshold

# Example usage

#input_dir = r"/content/drive/MyDrive/Datasets/DRIVE/training/mask"

input_dir = sys.argv[1] if len(sys.argv) > 1 else "/content/drive/MyDrive/Datasets/DRIVE/training/mask"
print(f"[INFO] Input Data Directory: {input_dir}")

if "test" in input_dir:
    thin_dir = './thin_vessels/test'
    thick_dir = './thick_vessels/test'
else:
    thin_dir = './thin_vessels/train'
    thick_dir = './thick_vessels/train'

os.makedirs(thin_dir, exist_ok=True)
os.makedirs(thick_dir, exist_ok=True)

for fname in sorted(os.listdir(input_dir)):
    if not fname.lower().endswith(('.gif', '.png')):
        continue

    path = os.path.join(input_dir, fname)
    img_pil = Image.open(path)
    img = np.array(img_pil)
    img = img > 128

    thin_mask, thick_mask, auto_diam, auto_minlen, ecc_thr, size_thr = mask_to_thin_thick(img)
    base, _ = os.path.splitext(fname)
    imsave(os.path.join(thin_dir, f'{base}_thin.png'), (thin_mask * 255).astype(np.uint8))
    imsave(os.path.join(thick_dir, f'{base}_thick.png'), (thick_mask * 255).astype(np.uint8))
    print(f'{fname}: thin px={thin_mask.sum()}, thick px={thick_mask.sum()}, '
          f'diam_thr={auto_diam:.2f}, min_len={auto_minlen}, ecc_thr={ecc_thr}, size_thr={size_thr}')


