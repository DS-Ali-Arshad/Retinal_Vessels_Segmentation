import os
import sys
import numpy as np
import cv2
from sklearn.utils import shuffle
import tensorflow as tf
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, ElasticTransform, GridDistortion, OpticalDistortion, CoarseDropout


W, H = 512, 512


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# def load_data(path):
#     """ X = Images and Y = masks """

#     train_x = sorted(glob(os.path.join(path, "training", "images", "*.tif")))
#     train_y = sorted(glob(os.path.join(path, "training", "mask", "*.gif")))

#     test_x = sorted(glob(os.path.join(path, "test", "images", "*.tif")))
#     test_y = sorted(glob(os.path.join(path, "test", "mask", "*.gif")))

#     return (train_x, train_y), (test_x, test_y)


def load_folder(path):
    return sorted(glob(path))

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def load_data(path, augmented=False):

    if augmented:
        train_folder = "train"
        test_folder = "test"
        img_subfolder = "image"
        mask_subfolder = "mask"
        img_ext = "*.png"
        mask_ext = "*.png"
    else:
        train_folder = "training"
        test_folder = "test"
        img_subfolder = "images"
        mask_subfolder = "mask"
        img_ext = "*.tif"
        mask_ext = "*.gif"

    train_x = sorted(glob(os.path.join(path, train_folder, img_subfolder, img_ext)))
    train_y = sorted(glob(os.path.join(path, train_folder, mask_subfolder, mask_ext)))

    test_x = sorted(glob(os.path.join(path, test_folder, img_subfolder, img_ext)))
    test_y = sorted(glob(os.path.join(path, test_folder, mask_subfolder, mask_ext)))

    return (train_x, train_y), (test_x, test_y)

def clahe_equalized(img):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    separately to each RGB channel to boost local contrast.
    """
    assert len(img.shape) == 3 and img.shape[2] == 3, "Input must be H×W×3 uint8"
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    out = np.empty_like(img)
    for c in range(3):
        out[:,:,c] = clahe.apply(img[:,:,c])
    return out


def augment_data(images, masks, save_path, augment=True):
    """
    On-disk augmentation: flips, elastic/grid/optical distortions,
    then CLAHE + resize → save to save_path/image and save_path/mask.
    """
    create_dir(os.path.join(save_path, "image"))
    create_dir(os.path.join(save_path, "mask"))

    for idx, (img_path, mask_path) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        name = os.path.splitext(os.path.basename(img_path))[0]
        x = cv2.imread(img_path, cv2.IMREAD_COLOR)
        y = imageio.mimread(mask_path)[0]

        if augment:
            transforms = [
                HorizontalFlip(p=0.7),
                VerticalFlip(p=0.7),
                ElasticTransform(p=1, alpha=120, sigma=120*0.05),
                GridDistortion(p=1),
                OpticalDistortion(p=1, distort_limit=0.05)
            ]
            X, Y = [x], [y]
            for aug in transforms:
                out = aug(image=x, mask=y)
                X.append(out["image"])
                Y.append(out["mask"])
        else:
            X, Y = [x], [y]

        for i, (xi, yi) in enumerate(zip(X, Y)):
            xi = clahe_equalized(xi)
            xi = cv2.resize(xi, (W, H))
            #yi = cv2.resize(yi, (W, H))
            yi = cv2.resize(yi, (W, H), interpolation=cv2.INTER_NEAREST)
            _, yi = cv2.threshold(yi, 127, 255, cv2.THRESH_BINARY) #optional

            suffix = f"_{i}" if len(X) > 1 else ""
            img_name  = f"{name}{suffix}.png"
            mask_name = f"{name}{suffix}.png"

            cv2.imwrite(os.path.join(save_path, "image", img_name), xi)
            cv2.imwrite(os.path.join(save_path, "mask",  mask_name), yi)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)              ## (512, 512, 1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(4)
    return dataset


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    """ Load the data """
    if len(sys.argv) >= 2:
        data_path = sys.argv[1]
        print(f"[INFO] Using custom data path: {data_path}")
    else:
        data_path = r"/content/drive/MyDrive/Datasets/DRIVE"
        print(f"[INFO] No path provided. Using default: {data_path}")
    
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Creating directories """
    create_dir("new_data/train/image")
    create_dir("new_data/train/mask")
    create_dir("new_data/test/image")
    create_dir("new_data/test/mask")

    augment_data(train_x, train_y, "new_data/train/", augment=True)
    augment_data(test_x, test_y, "new_data/test/", augment=False)