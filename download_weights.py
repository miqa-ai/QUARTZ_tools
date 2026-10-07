"""
Download the QUARTZ model weights from Zenodo (https://zenodo.org/records/23188999) into the tool directories.

Usage:
    python download_weights.py [--force]
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

RECORD_ID = '23188999'
API_URL = f'https://zenodo.org/api/records/{RECORD_ID}'
ROOT = Path(__file__).resolve().parent

# Weight file -> tool directory it belongs to
DESTINATIONS = {
    'av_seg_best_model2_UKBB.h5': 'segmentation',
    'od_seg_UKBB.h5': 'OD',
    'imgQ_UKBB_DUAL.h5': 'ImageQuality',
}


def md5sum(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def download(url, dest, size):
    tmp = dest.with_suffix(dest.suffix + '.part')
    with urllib.request.urlopen(url) as r, open(tmp, 'wb') as f:
        done = 0
        while block := r.read(1 << 20):
            f.write(block)
            done += len(block)
            print(f'\r  {done / 1e6:7.1f} / {size / 1e6:.1f} MB', end='', flush=True)
    print()
    tmp.replace(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--force', action='store_true', help='re-download even if a valid file already exists')
    args = parser.parse_args()

    with urllib.request.urlopen(API_URL) as r:
        record = json.load(r)

    ok = True
    for entry in record['files']:
        name = entry['key']
        if name not in DESTINATIONS:
            continue
        dest = ROOT / DESTINATIONS[name] / name
        md5 = entry['checksum'].split(':', 1)[-1]
        if dest.exists() and not args.force and md5sum(dest) == md5:
            print(f'{dest.relative_to(ROOT)}: already present')
            continue
        print(f'Downloading {name} -> {dest.relative_to(ROOT)}')
        download(entry['links']['self'], dest, entry['size'])
        if md5sum(dest) != md5:
            print(f'  ERROR: checksum mismatch for {name}', file=sys.stderr)
            ok = False

    missing = [n for n in DESTINATIONS if n not in {e['key'] for e in record['files']}]
    if missing:
        print(f'ERROR: not found in Zenodo record: {", ".join(missing)}', file=sys.stderr)
        ok = False
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
