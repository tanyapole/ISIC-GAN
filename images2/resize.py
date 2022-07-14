import torchvision.transforms as TF
import torchvision.transforms.functional as TFF
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
from tqdm.auto import tqdm


def _get_new_shape(img):
    H, W = np.array(img).shape[:2]
    H_new, W_new = 512, round(512 * W / H)
    return H_new, W_new

def resize_image(img):
    H_new, W_new = _get_new_shape(img)
    mode = TF.InterpolationMode.BILINEAR
    return TFF.resize(img, (H_new, W_new), interpolation=mode)

def resize_mask(img):
    H_new, W_new = _get_new_shape(img)
    mode = TF.InterpolationMode.NEAREST
    return TFF.resize(img, (H_new, W_new), interpolation=mode)

def resize_folder(source, target, rtype):
    source ,target = Path(source), Path(target)
    print('source is ', source.absolute())
    target.mkdir(exist_ok=False, parents=True)
    skipped = []
    for pt in tqdm(list(source.iterdir())):
        if pt.suffix in ['.jpg', '.png']:
            image = Image.open(pt)
            if rtype == 'image':
                image = resize_image(image)
            else:
                image = resize_mask(image)
            targ_name = pt.name[:-len(pt.suffix)] + '.png'
            image.save(target / targ_name)
        else:
            skipped.append(pt.name)
    print(f'Skipped files: {", ".join(skipped)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('resize')
    parser.add_argument('--source', type=str, help='absolute path to source folder', required=True)
    parser.add_argument('--target', type=str, help='absolute path to target folder', required=True)
    parser.add_argument('--type', choices=['image', 'mask'], help='resize type', required=True)
    args = parser.parse_args()
    print(args)
    print('Start resizing...')
    resize_folder(args.source ,args.target, args.type)
    print('Resizing ended')
    


