import torch
from PIL import Image
import torchvision.transforms.functional as TFF
import numpy as np
from enum import Enum
import argparse
from pathlib import Path
from tqdm.auto import tqdm


class Attributes(Enum):
    pigm = 'pigment_network'
    neg = 'negative_network'
    streaks = 'streaks'
    milia = 'milia_like_cyst'
    globules = 'globules'
    
_root = Path('.')

def _read_mask(pt):
    image = Image.open(pt)
    return (np.array(image) > 0).astype(np.uint8)

def combine_masks(segm_mask, attr_masks):
    rgb = segm_mask + 1
    for i, attr in enumerate(Attributes):
        rgb[attr_masks[attr] == 1] = i+3
    return rgb

def _read_segm_mask(ID):
    return _read_mask(_root / 'seg_512p' / f'{ID}_segmentation.png')

def _read_attr_masks(ID):
    return {
        attr: _read_mask(_root / 'attribute_512p' / f'{ID}_attribute_{attr.value}.png')
        for attr in Attributes
    }

def _get_IDs():
    root = Path('images_512p')
    L = root.iterdir()
    return [l.name[:-len(l.suffix)] for l in L]

def combine_all():
    _target_folder = _root / 'combined_masks'
    _target_folder.mkdir(exist_ok=False)
    IDs = _get_IDs()
    for ID in IDs:
        segm = _read_segm_mask(ID)
        attrs = _read_attr_masks(ID)
        combined = combine_masks(segm, attrs)
        Image.fromarray(combined).save(_target_folder / f'{ID}_semantic.png')

if __name__ == '__main__':
    combine_all()