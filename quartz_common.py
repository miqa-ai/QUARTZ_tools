"""
Shared helpers for the QUARTZ inference scripts (image discovery/loading, FOV masks, model loading, CSV output).
"""
import csv
import os
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # silence TF info/warning logs

import cv2
import numpy as np
import tensorflow as tf

REPO_DIR = Path(__file__).resolve().parent

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

# All models were developed on images at the UK Biobank resolution (H x W). Images of any other size are resized to
# this "native" resolution first (as performed by readImage.m in the original MATLAB pipeline).
NATIVE_H, NATIVE_W = 1536, 2048


def setup_gpu():
    """Allocate GPU memory on demand rather than grabbing all of it."""
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def load_model(path):
    """Load a Keras .h5 model for inference only (compile=False avoids needing the custom training losses/metrics)."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f'Model weights not found: {path}\nRun "python download_weights.py" from the repository root first.')
    return tf.keras.models.load_model(str(path), compile=False)


def list_images(img_dir):
    """Sorted list of retinal image files in img_dir (non-recursive)."""
    img_dir = Path(img_dir)
    if not img_dir.is_dir():
        raise SystemExit(f'Input directory not found: {img_dir}')
    files = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith('.'))
    if not files:
        raise SystemExit(f'No images ({", ".join(sorted(IMG_EXTS))}) found in {img_dir}')
    return files


def read_image(img_path):
    """Read an RGB image as uint8 tensor (H, W, 3). Uses TF decoders where possible (as in development), else OpenCV."""
    try:
        return tf.io.decode_image(tf.io.read_file(str(img_path)), channels=3, expand_animations=False)
    except (tf.errors.InvalidArgumentError, tf.errors.UnimplementedError):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f'Cannot read image {img_path}')
        if img.dtype != np.uint8:
            img = (img / np.iinfo(img.dtype).max * 255).round().astype(np.uint8)
        return tf.convert_to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def to_native(image):
    """Resize image to the native development resolution if needed. Returns float32 with values in [0, 255]."""
    image = tf.cast(image, tf.float32)
    if image.shape[:2] != (NATIVE_H, NATIVE_W):
        image = tf.image.resize(image, [NATIVE_H, NATIVE_W], method='bicubic', antialias=True)  # same interpolation as used in development
    return image


def compute_fov_mask(image, threshold=15):
    """
    Estimate the circular field-of-view (FOV) mask of a fundus image (float [0, 255], H x W x 3).
    The models only use the FOV extent to crop a square region around the retina, so a simple intensity threshold
    followed by keeping the largest filled connected component is sufficient.
    Returns uint8 array (H, W, 1) with values 0/255.
    """
    img = np.asarray(image)
    gray = cv2.GaussianBlur(img.mean(axis=2).astype(np.float32), (0, 0), 5)
    mask = (gray > threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError('Could not detect the retinal field of view (image too dark?)')
    fov = np.zeros_like(mask)
    cv2.drawContours(fov, [max(contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
    return fov[..., np.newaxis]


def get_fov_mask(image, img_path, mask_dir=None):
    """FOV mask for image: read <mask_dir>/<image stem>.png if mask_dir is given, otherwise compute it."""
    if mask_dir is None:
        return compute_fov_mask(image)
    mask_path = Path(mask_dir) / (Path(img_path).stem + '.png')
    mask = tf.io.decode_png(tf.io.read_file(str(mask_path)), channels=1)
    if mask.shape[:2] != image.shape[:2]:
        mask = tf.image.resize(mask, image.shape[:2], method='nearest')
    return mask.numpy()


def squ_crop(image, fov_mask):
    """Square crop around the FOV (same function as used in training)."""
    fov_idx = tf.where(fov_mask)  # index of non-zeros
    fov_idx_h = fov_idx[:, 1]  # column index
    fov_idx_v = fov_idx[:, 0]  # row index

    h1 = tf.reduce_min(fov_idx_h)  # first point horizontally
    h2 = tf.reduce_max(fov_idx_h)  # last point horizontally
    diam = tf.cast(h2 - h1, tf.int32)

    v1 = tf.reduce_min(fov_idx_v)  # first point vertically
    v2 = tf.reduce_max(fov_idx_v)  # last point vertically

    image_crop = image[v1:v2, h1:h2, :]
    image_crop = tf.image.resize_with_crop_or_pad(image_crop, diam, diam)  # ensures square (used diam from horz), hence may further crop or pad vertically. Done centrally.

    return image_crop, h1, h2, v1, v2


def uncrop(pred_mask, h1, h2, v1, v2, channels, dtype):
    """Place a square-cropped prediction back into an array of the native resolution (reverse of squ_crop)."""
    # tf.image.resize_with_crop_or_pad pads/crops centrally, so applying it again exactly reverses the vertical crop/pad
    pred_mask = tf.image.resize_with_crop_or_pad(pred_mask, tf.cast(v2 - v1, tf.int32), tf.cast(h2 - h1, tf.int32))
    out = np.zeros((NATIVE_H, NATIVE_W, channels), dtype=dtype)
    out[v1:v2, h1:h2, :pred_mask.shape[-1]] = pred_mask
    return out


def write_png(path, mask):
    """Write uint8 mask (H, W, C) as PNG."""
    tf.io.write_file(str(path), tf.io.encode_png(tf.convert_to_tensor(mask, dtype=tf.uint8)))


class CsvWriter:
    """Writes one row per image, flushing after each so partial results survive an interrupted run."""

    def __init__(self, path, fieldnames):
        self.path = Path(path)
        self.file = open(self.path, 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames, extrasaction='ignore')
        self.writer.writeheader()

    def write(self, row):
        self.writer.writerow({k: ('' if v is None or (isinstance(v, float) and np.isnan(v)) else v) for k, v in row.items()})
        self.file.flush()

    def close(self):
        self.file.close()


def fmt_float(x, ndigits=4):
    return float('nan') if x is None or np.isnan(x) else round(float(x), ndigits)
