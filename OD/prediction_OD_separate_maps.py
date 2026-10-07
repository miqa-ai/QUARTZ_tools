"""
Optic disc (OD) and optic cup (OC) segmentation of retinal fundus images.

Produces OD/OC segmentation maps as PNG (R = disc, G = cup) at the native resolution of 2048 x 1536.

Usage:
    python OD/prediction_OD_separate_maps.py <image_dir> [--out-dir od_seg]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quartz_common as qc  # noqa: E402  (imported first: sets TF logging before tensorflow loads)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent / 'od_seg_UKBB.h5'

IMG_SIZE = 512
T_OD = 0.43  # OD threshold
T_OC = 0.36  # OC threshold


def mathematical_morphology(pred_mask):
    """Paper by Liu applied closing with 7 x 7 circular SE on images of 480x480 and 560x560. FYI, the seg outputs look
    great without this, but let's see if this makes a difference to results.

    FYI, made zero difference as there are no gaps/holes to fill and the surface was already very smooth.

    Some papers also apply ellipse fitting, I am not too drawn to this as the GTS are never fully elliptical.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    for c in range(2):  # OD, OC
        pred_mask[:, :, c] = cv2.morphologyEx(pred_mask[:, :, c].astype('uint8'), cv2.MORPH_CLOSE, kernel)
    return pred_mask


def image_prediction(model, img, fov_mask):
    """
    OD and OC segmentation creating separate maps for OD and OC.

    img: float32 [0, 1] image at native resolution (1536 x 2048 x 3). fov_mask: (H, W, 1) FOV mask.
    Returns pred_mask uint8 H x W x 3 with channels OD, OC, 0 and values 0/1.
    """
    img_native_crop, h1, h2, v1, v2 = qc.squ_crop(img, fov_mask)  # e.g might give 1564,1564,3
    img = tf.image.resize(img_native_crop, [IMG_SIZE, IMG_SIZE], antialias=True)  # default interpolation is bilinear. # 512, 512, 3

    pred_mask = model(img[tf.newaxis, ...], training=False)  # 1,512,512,2
    # Resize probability map to native cropped resolution
    img_size_N = img_native_crop.shape[0]
    pred_mask = tf.image.resize(pred_mask, [img_size_N, img_size_N], antialias=True)  # e.g. might give 1,1564,1564,2. Default interpolation is bilinear.
    pred_mask = pred_mask.numpy()[0]

    # THRESHOLDING, done separately, so seg map for each class
    pred_mask = np.stack([pred_mask[:, :, 0] >= T_OD, pred_mask[:, :, 1] >= T_OC], axis=-1).astype(np.uint8)

    # Does post-processing help? (made no difference, see above)
    # pred_mask = mathematical_morphology(pred_mask)

    # Place the pred_mask (native cropped res) into an array of zeros of the native resolution prior to crop.
    # 3 channels so it is ready to be viewed/saved as an image.
    return qc.uncrop(pred_mask, h1, h2, v1, v2, channels=3, dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description='Optic disc / cup segmentation of retinal images.')
    parser.add_argument('image_dir', help='directory with retinal images (png, jpg, tif, bmp)')
    parser.add_argument('--out-dir', default='od_seg', help='where to save the OD/OC maps as PNG (R=disc, G=cup) [default: ./od_seg]')
    parser.add_argument('--mask-dir', help='optional directory with FOV masks (<image name>.png); computed automatically if omitted')
    parser.add_argument('--model', default=str(MODEL_PATH), help='model weights (.h5)')
    args = parser.parse_args()

    qc.setup_gpu()
    files = qc.list_images(args.image_dir)
    model = qc.load_model(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_ok = 0
    for n, img_path in enumerate(files, 1):
        try:
            img = qc.to_native(qc.read_image(img_path))
            fov_mask = qc.get_fov_mask(img, img_path, args.mask_dir)
            od_mask = image_prediction(model, img / 255.0, fov_mask)
            qc.write_png(out_dir / (img_path.stem + '.png'), od_mask * 255)  # * 255 so viewable outside of python
            status = 'ok'
            n_ok += 1
        except Exception as e:  # keep going on bad images
            status = f'error: {e}'
        print(f'[{n}/{len(files)}] {img_path.name}: {status}', flush=True)
    print(f'Done: {n_ok}/{len(files)} images in {time.time() - t0:.1f}s. Maps: {out_dir}/')

if __name__ == '__main__':
    main()
