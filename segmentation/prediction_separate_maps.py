"""
Artery / vein / vessel segmentation of retinal fundus images.

U-Net with ConvNeXt blocks producing separate binary maps for artery, vein and vessel.

Usage:
    python segmentation/prediction_separate_maps.py <image_dir> [--out-dir av_seg] [--csv av_segmentation.csv]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quartz_common as qc  # noqa: E402  (imported first: sets TF logging before tensorflow loads)

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent / 'av_seg_best_model2_UKBB.h5'

IMG_SIZE = 1024
# Thresholds
T_ART = 0.38
T_VEI = 0.42
T_VES = 0.42

CSV_FIELDS = ['filename', 'artery_px', 'vein_px', 'vessel_px', 'fov_px', 'vessel_density', 'artery_vein_ratio',
              'seg_map', 'status']


def image_prediction(model, img, fov_mask):
    """
    A/V segmentation creating a separate map for artery, vein, vessel.

    img: float32 [0, 1] image at native resolution (1536 x 2048 x 3). fov_mask: (H, W, 1) FOV mask.
    Returns binary map (uint8 H x W x 3, values 0/1, channels = artery, vein, vessel) and the probability map
    (float32 H x W x 3), both at native resolution.
    """
    img_native_crop, h1, h2, v1, v2 = qc.squ_crop(img, fov_mask)
    img = tf.image.resize(img_native_crop, [IMG_SIZE, IMG_SIZE], antialias=True)  # default interpolation is bilinear.

    pred_mask = model(img[tf.newaxis, ...], training=False)
    # Resize probability map to native cropped resolution
    img_size_N = img_native_crop.shape[0]
    pred_mask = tf.image.resize(pred_mask, [img_size_N, img_size_N], antialias=True)  # Default interpolation is bilinear.
    prob_mask = pred_mask.numpy()[0]

    # THRESHOLDING, each map separately
    seg_mask = np.stack([prob_mask[:, :, 0] >= T_ART,
                         prob_mask[:, :, 1] >= T_VEI,
                         prob_mask[:, :, 2] >= T_VES], axis=-1).astype(np.uint8)

    # Place the predictions (native cropped res) back into an array of the native resolution prior to crop
    seg_final = qc.uncrop(seg_mask, h1, h2, v1, v2, channels=3, dtype=np.uint8)
    prob_final = qc.uncrop(prob_mask, h1, h2, v1, v2, channels=3, dtype=np.float32)
    return seg_final, prob_final


def segment_file(model, img_path, mask_dir=None):
    """Load an image file, resize to native resolution and segment it. Returns (seg_map, prob_map, fov_mask)."""
    img = qc.to_native(qc.read_image(img_path))
    fov_mask = qc.get_fov_mask(img, img_path, mask_dir)
    seg, prob = image_prediction(model, img / 255.0, fov_mask)
    return seg, prob, fov_mask


def load_av_model(path=MODEL_PATH):
    return qc.load_model(path)


def main():
    parser = argparse.ArgumentParser(description='Artery/vein/vessel segmentation of retinal images.')
    parser.add_argument('image_dir', help='directory with retinal images (png, jpg, tif, bmp)')
    parser.add_argument('--out-dir', default='av_seg',
                        help='where to save the segmentation maps as PNG (R=artery, G=vein, B=vessel) [default: ./av_seg]')
    parser.add_argument('--csv', default='av_segmentation.csv', help='output CSV [default: ./av_segmentation.csv]')
    parser.add_argument('--mask-dir', help='optional directory with FOV masks (<image name>.png); computed automatically if omitted')
    parser.add_argument('--save-prob', action='store_true', help='also save probability maps as <image name>.h5 (dataset "data")')
    parser.add_argument('--model', default=str(MODEL_PATH), help='model weights (.h5)')
    args = parser.parse_args()

    qc.setup_gpu()
    files = qc.list_images(args.image_dir)
    model = load_av_model(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_out = qc.CsvWriter(args.csv, CSV_FIELDS)
    t0 = time.time()
    for n, img_path in enumerate(files, 1):
        row = {'filename': img_path.name}
        try:
            seg, prob, fov_mask = segment_file(model, img_path, args.mask_dir)
            seg_path = out_dir / (img_path.stem + '.png')
            qc.write_png(seg_path, seg * 255)  # * 255 so viewable outside of python
            if args.save_prob:
                with h5py.File(out_dir / (img_path.stem + '.h5'), 'w') as hdf_file:  # dims are reversed when opened in Matlab
                    hdf_file.create_dataset('data', data=prob, compression='gzip')

            art, vei, ves = (int(seg[:, :, c].sum()) for c in range(3))
            fov_px = int((fov_mask > 0).sum())
            row.update(artery_px=art, vein_px=vei, vessel_px=ves, fov_px=fov_px,
                       vessel_density=qc.fmt_float(ves / fov_px if fov_px else np.nan),
                       artery_vein_ratio=qc.fmt_float(art / vei if vei else np.nan),
                       seg_map=str(seg_path), status='ok')
        except Exception as e:  # keep going on bad images, report in CSV
            row['status'] = f'error: {e}'
        csv_out.write(row)
        print(f'[{n}/{len(files)}] {img_path.name}: {row["status"]}', flush=True)
    csv_out.close()
    print(f'Done: {len(files)} images in {time.time() - t0:.1f}s. Results: {args.csv}, maps: {out_dir}/')


if __name__ == '__main__':
    main()
