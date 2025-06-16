import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def combine_vessels(thin_dir, thick_dir, combined_dir):
    os.makedirs(combined_dir, exist_ok=True)
    thin_files = sorted([f for f in os.listdir(thin_dir) if f.endswith('_thin.png')])

    for thin_file in thin_files:
        base_name = thin_file.replace('_thin.png', '')
        thick_file = f"{base_name}_thick.png"

        thin_path = os.path.join(thin_dir, thin_file)
        thick_path = os.path.join(thick_dir, thick_file)
        combined_path = os.path.join(combined_dir, f"{base_name}_combined.png")

        if not os.path.exists(thick_path):
            print(f"Skipping {base_name}: thick image not found.")
            continue

        thin_img = np.array(Image.open(thin_path).convert("L")) > 0
        thick_img = np.array(Image.open(thick_path).convert("L")) > 0

        combined = (thin_img | thick_img).astype(np.uint8) * 255
        Image.fromarray(combined).save(combined_path)

    print(f"Combined vessel images saved to: {combined_dir}")


# === Main execution ===
original_path = sys.argv[1] if len(sys.argv) > 1 else "/content/drive/MyDrive/Datasets/DRIVE/training/mask"
thin_dir = "./thin_vessels"
thick_dir = "./thick_vessels"
combined_dir = "./combined_vessels"
output_dir = "./Original_Vs_Combined_Results"
os.makedirs(output_dir, exist_ok=True)

combine_vessels(thin_dir, thick_dir, combined_dir)

filenames = [f"{i}_manual1" for i in range(21, 41)]

for name in filenames:
    original_path_full = os.path.join(original_path, f"{name}.gif")
    combined_path_full = os.path.join(combined_dir, f"{name}_combined.png")

    if not os.path.exists(original_path_full) or not os.path.exists(combined_path_full):
        print(f"Skipping {name}: missing original or combined image.")
        continue

    original_img = Image.open(original_path_full).convert("RGB")
    combined_img = Image.open(combined_path_full).convert("RGB")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Original Mask')
    axes[0].axis('off')

    axes[1].imshow(combined_img, cmap='gray')
    axes[1].set_title('Combined Vessels')
    axes[1].axis('off')

    output_path = os.path.join(output_dir, f"{name}_comparison.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

print(f"Side-by-side comparisons saved to: {output_dir}")
