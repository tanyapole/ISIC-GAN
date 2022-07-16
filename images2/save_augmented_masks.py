from pathlib import Path
import pandas as pd
import os

from PIL import Image
import numpy as np
import pickle
from tqdm.auto import tqdm
import torchvision.transforms.functional as TFF

def get_augmented(img, idx):
    if idx == 0:
        return img
    elif idx == 1:
        return TFF.hflip(img)
    elif idx == 2:
        return TFF.vflip(img)
    elif idx == 3:
        return TFF.vflip(TFF.hflip(img))
    else:
        raise Exception(f"Unsupported augmentation idx={idx}")

def main():
    masks_fldr = Path('attribute_512p')

    targ_fldr = Path('attribute_512p_augmented')
    targ_fldr.mkdir(exist_ok=False)

    for pt in tqdm(list(masks_fldr.iterdir())):
        for j in range(4):
            image = Image.open(pt)
            image = get_augmented(image, j)
            targ_name = pt.name[:-len(pt.suffix)] + f'_v{j}' + '.png'
            image.save(targ_fldr / targ_name)

if __name__ == '__main__':
    main()