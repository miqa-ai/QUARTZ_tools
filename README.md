# QUARTZ tools

Deep learning tools for automated analysis of colour retinal fundus images. These tools were developed as part of the
research work published in Welikala et al., *"Artificial intelligence-enabled retinal vasculometry at scale utilizing
the UK Biobank, CLSA, and NEL DESP datasets"*, IEEE EMBS BHI 2024 ([doi:10.1109/BHI62660.2024.10913687](https://doi.org/10.1109/BHI62660.2024.10913687));
see [Citation](#citation). There are three tools:

| Tool | Directory | What it does | CSV output |
|---|---|---|---|
| **A/V segmentation** | `segmentation/` | U-Net with ConvNeXt blocks segmenting separate **artery**, **vein** and **vessel** maps | `av_segmentation.csv` + maps in `av_seg/` |
| **Optic disc** | `OD/` | Segments the **optic disc** and **optic cup** | maps in `od_seg/` (no CSV) |
| **Image quality** | `ImageQuality/` | Dual-input CNN (image + A/V segmentation) giving a **quality score** in [0, 1] (high = good) | `image_quality.csv` |

## Installation

Requires Python 3.9–3.11. The models are Keras 2 `.h5` files, so they need TensorFlow 2.11–2.15. A GPU is
recommended but not required.

```bash
git clone <this repo> QUARTZ && cd QUARTZ
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt     # for GPU support on Linux: pip install "tensorflow[and-cuda]>=2.11,<2.16"
python download_weights.py          # downloads the model weights (~205 MB) from https://zenodo.org/records/23188999
```

## Usage

Each tool takes a directory of retinal images (`.png`, `.jpg`, `.tif`, `.bmp`) and writes its results into the
**current directory**: a CSV file with one row per image (A/V segmentation, image quality) and/or segmentation maps.
Images that cannot be processed are listed with `status = error: ...`.

```bash
# 1) Artery / vein / vessel segmentation -> av_segmentation.csv and maps in ./av_seg/
python segmentation/prediction_separate_maps.py /path/to/images

# 2) Optic disc / cup -> maps in ./od_seg/
python OD/prediction_OD_separate_maps.py /path/to/images

# 3) Image quality -> image_quality.csv (reuses the maps from step 1)
python ImageQuality/prediction_dual_input.py /path/to/images --seg-dir av_seg
```

Without `--seg-dir`, the image quality tool runs the A/V segmentation itself, so it can also be used on its own.
Run any script with `-h` to see all options (`--csv`, `--out-dir`, `--mask-dir`, `--model`, ...).

**Output columns**

- `av_segmentation.csv`: `artery_px`, `vein_px`, `vessel_px` (segmented pixels), `fov_px` (field-of-view pixels),
  `vessel_density` (= vessel_px / fov_px), `artery_vein_ratio` (= artery_px / vein_px), `seg_map` (path to the map PNG).
  Maps are RGB PNGs: R = artery, G = vein, B = vessel. `--save-prob` also saves the probability maps (`.h5`).
- `od_seg/<image name>.png`: optic disc / cup maps, R = disc (whole disc incl. cup), G = cup.
- `image_quality.csv`: `quality_score` (1 − probability of an inadequate image), `adequate` (1 if score > 0.66).

**Notes**

- The models were developed on UK Biobank images (2048 × 1536). Images of other sizes are resized to 2048 × 1536 first,
  and all pixel measurements refer to this resized image.
- The tools crop a square around the retinal field of view (FOV). The FOV mask is estimated automatically. You can
  supply your own binary masks with `--mask-dir` (files named `<image name>.png`).

## Citation

If you use these tools, please cite:

> R. Welikala, J. Fajtl, G. Johnson, F. Rahman, R. Podoleanu, P. Remagnino, E. E. Freeman, R. Chambers, L. Bolter,
> J. Anderson, A. Olvera-Barrios, A. Warwick, P. J. Foster, R. Shakespeare, R. Ganguly, C. Egan, A. Tufail, C. G. Owen,
> A. R. Rudnicka, S. A. Barman, "Artificial intelligence-enabled retinal vasculometry at scale utilizing the UK Biobank,
> CLSA, and NEL DESP datasets," *2024 IEEE EMBS International Conference on Biomedical and Health Informatics (BHI)*,
> IEEE, 2024. doi: [10.1109/BHI62660.2024.10913687](https://doi.org/10.1109/BHI62660.2024.10913687)

```bibtex
@inproceedings{welikala2024quartz,
  title     = {Artificial intelligence-enabled retinal vasculometry at scale utilizing the {UK Biobank}, {CLSA}, and {NEL DESP} datasets},
  author    = {Welikala, Roshan and Fajtl, Jiri and Johnson, Gordon and Rahman, Farzana and Podoleanu, Razvan and
               Remagnino, Paolo and Freeman, Ellen E. and Chambers, Ryan and Bolter, Louis and Anderson, John and
               Olvera-Barrios, Abraham and Warwick, Alasdair and Foster, Paul J. and Shakespeare, Royce and Ganguly, Rahul and
               Egan, Catherine and Tufail, Adnan and Owen, Christopher G. and Rudnicka, Alicja R. and Barman, Sarah A.},
  booktitle = {2024 IEEE EMBS International Conference on Biomedical and Health Informatics (BHI)},
  publisher = {IEEE},
  year      = {2024},
  doi       = {10.1109/BHI62660.2024.10913687}
}
```

Publication page: <https://researchinnovation.kingston.ac.uk/en/publications/artificial-intelligence-enabled-retinal-vasculometry-at-scale-uti-3/>
Model weights: <https://zenodo.org/records/23188999>
