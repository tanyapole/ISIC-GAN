from PIL import Image
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

def _read_segm_mask(ID, is_bbox):
    sub_fldr = 'bboxes_seg_512p' if is_bbox else 'seg_512p'
    return _read_mask(_root / sub_fldr / f'{ID}_segmentation.png')

def _read_attr_masks(ID, is_bbox):
    sub_fldr = 'bboxes_attribute_512p' if is_bbox else 'attribute_512p'
    return {
        attr: _read_mask(_root / sub_fldr / f'{ID}_attribute_{attr.value}.png')
        for attr in Attributes
    }

def _get_IDs():
    root = Path('images_512p')
    L = root.iterdir()
    return [l.name[:-len(l.suffix)] for l in L]

def combine_all(is_bbox:bool):
    sub_fldr = 'bboxes_combined_masks' if is_bbox else 'combined_masks'
    _target_folder = _root / sub_fldr
    _target_folder.mkdir(exist_ok=False)
    IDs = _get_IDs()
    for ID in tqdm(IDs):
        segm = _read_segm_mask(ID, is_bbox)
        attrs = _read_attr_masks(ID, is_bbox)
        combined = combine_masks(segm, attrs)
        Image.fromarray(combined).save(_target_folder / f'{ID}_semantic.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('combine mutlilabel masks into multiclass condition')
    parser.add_argument('--bbox', action='store_true')
    args = parser.parse_args()
    print(args)
    combine_all(args.bbox)