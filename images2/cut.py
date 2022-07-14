from PIL import Image
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import argparse


def get_ID(filepath):
    return '_'.join(filepath.name.split('_')[:2])

def get_left_right(orig_shape, synth_shape):
    assert (orig_shape[0] == synth_shape[0]), 'image height mismatch'
    diff = synth_shape[1] - orig_shape[1]
    left = diff // 2
    right = diff - left
    return left, right


def main(synthesized_fldr, target_fldr):
    before_pad_fldr = Path('images_512p')
    synthesized_fldr = Path(synthesized_fldr)
    target_fldr = Path(target_fldr)
    target_fldr.mkdir(exist_ok=False, parents=True)

    for filepath in tqdm(list(synthesized_fldr.iterdir())):
        if filepath.name.endswith('_semantic_synthesized_image.png'):
            ID = get_ID(filepath)
            original_path = before_pad_fldr / f'{ID}.png'
            original_shape = np.array(Image.open(original_path)).shape
            synthesized_image = np.array(Image.open(filepath))
            left, right = get_left_right(original_shape, synthesized_image.shape)
            cut_image = synthesized_image[:, left:-right]
            Image.fromarray(cut_image).save(target_fldr / filepath.name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('cut synthesized images')
    parser.add_argument('--source', type=str, help='absolute path to source folder', required=True)
    parser.add_argument('--target', type=str, help='absolute path to target folder', required=True)
    args = parser.parse_args()
    print(args)
    print('Start cutting...')
    main(args.source ,args.target)
    print('Cutting ended')