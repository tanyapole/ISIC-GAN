import torch
from PIL import Image
import torchvision.transforms.functional as TFF
import numpy as np
from enum import Enum
import argparse
from pathlib import Path
from tqdm.auto import tqdm


def pad(img):
    diff = 1024 - np.array(img).shape[1]
    left = diff // 2
    right = diff - left
    diff, left, right
    return TFF.pad(img, padding=(left, 0, right, 0), fill=0)

def pad_folder(source, target):
    source ,target = Path(source), Path(target)
    print('source is ', source.absolute())
    target.mkdir(exist_ok=False, parents=True)
    skipped = []
    for pt in tqdm(list(source.iterdir())):
        if pt.suffix in ['.jpg', '.png']:
            image = pad(Image.open(pt))
            targ_name = pt.name[:-len(pt.suffix)] + '.png'
            image.save(target / targ_name)
        else:
            skipped.append(pt.name)
    print(f'Skipped files: {", ".join(skipped)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('resize')
    parser.add_argument('--source', type=str, help='absolute path to source folder', required=True)
    parser.add_argument('--target', type=str, help='absolute path to target folder', required=True)
    args = parser.parse_args()
    print(args)
    print('Start padding...')
    pad_folder(args.source ,args.target)
    print('Padding ended')