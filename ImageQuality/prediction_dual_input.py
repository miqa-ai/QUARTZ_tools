"""
Image quality assessment of retinal fundus images.

Dual-input CNN taking the retinal image and its artery/vein/vessel segmentation (from segmentation/). Outputs a
quality score in [0, 1], where high means good. Images with a score <= 0.66 are considered inadequate.

Usage:
    python ImageQuality/prediction_dual_input.py <image_dir> [--seg-dir av_seg] [--csv image_quality.csv]

If --seg-dir is not given, the A/V segmentation is computed on the fly with the segmentation model.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'segmentation'))
import quartz_common as qc  # noqa: E402  (imported first: sets TF logging before tensorflow loads)

import tensorflow as tf  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent / 'imgQ_UKBB_DUAL.h5'

IMG_SIZE = 384
QUALITY_THRESHOLD = 0.66  # quality_score <= threshold -> inadequate (equivalent to P(inadequate) >= 0.34)

CSV_FIELDS = ['filename', 'quality_score', 'adequate', 'status']


def prediction(model, image, seg):
    """
    Image Quality with dual input of retinal image and av segmentation.

    image: native resolution image (1536 x 2048 x 3) with values in [0, 255].
    seg: A/V segmentation map (H x W x 3, uint8 values 0/255) as written by segmentation/prediction_separate_maps.py.
    Returns quality score (1 - probability of being inadequate).
    """
    # class_names = ['Adequate', 'Inadequate']
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE], antialias=True)  # default interpolation is bilinear, outputs float32
    seg = tf.image.resize(seg, [IMG_SIZE, IMG_SIZE], method='nearest')  # nearest retains seg integer values
    seg = tf.cast(seg, tf.float32)  # also get seg in float ready for the CNN

    pred = model((image[tf.newaxis, ...], seg[tf.newaxis, ...]), training=False)  # image and seg each become 1,384,384,3, output 1,1
    probability = float(pred.numpy()[0][0])  # probability of being inadequate

    # Flip the score so that a high value means a good image, will be easier for clinicians to understand.
    # The threshold of >=0.34 to exclude now corresponds to a threshold of <=0.66 to exclude.
    return 1 - probability


def main():
    parser = argparse.ArgumentParser(description='Image quality assessment of retinal images.')
    parser.add_argument('image_dir', help='directory with retinal images (png, jpg, tif, bmp)')
    parser.add_argument('--seg-dir', help='directory with A/V segmentation maps (<image name>.png) from '
                                          'segmentation/prediction_separate_maps.py; computed on the fly if omitted')
    parser.add_argument('--csv', default='image_quality.csv', help='output CSV [default: ./image_quality.csv]')
    parser.add_argument('--mask-dir', help='optional directory with FOV masks (only used when segmenting on the fly)')
    parser.add_argument('--model', default=str(MODEL_PATH), help='image quality model weights (.h5)')
    args = parser.parse_args()

    qc.setup_gpu()
    files = qc.list_images(args.image_dir)
    model = qc.load_model(args.model)
    av_model = None
    if args.seg_dir is None:
        import prediction_separate_maps as av  # segmentation/prediction_separate_maps.py
        av_model = av.load_av_model()

    csv_out = qc.CsvWriter(args.csv, CSV_FIELDS)
    t0 = time.time()
    for n, img_path in enumerate(files, 1):
        row = {'filename': img_path.name}
        try:
            image = qc.to_native(qc.read_image(img_path))
            if av_model is None:
                seg = tf.io.decode_png(tf.io.read_file(str(Path(args.seg_dir) / (img_path.stem + '.png'))), channels=3)
            else:
                seg, _, _ = av.segment_file(av_model, img_path, args.mask_dir)
                seg = seg * 255  # same values as the saved PNG maps
            score = prediction(model, image, seg)
            row.update(quality_score=qc.fmt_float(score), adequate=int(score > QUALITY_THRESHOLD), status='ok')
        except Exception as e:  # keep going on bad images, report in CSV
            row['status'] = f'error: {e}'
        csv_out.write(row)
        print(f'[{n}/{len(files)}] {img_path.name}: {row["status"]}', flush=True)
    csv_out.close()
    print(f'Done: {len(files)} images in {time.time() - t0:.1f}s. Results: {args.csv}')


if __name__ == '__main__':
    main()
