import argparse
import pickle
from pathlib import Path
import shutil
from tqdm.auto import tqdm


def _read_ids(pt):
    with open(pt, 'rb') as f:
        return pickle.load(f)

def _handle(target_folder, phase):
    IDs = _read_ids(f'{phase}_ids.pkl')
    img_fldr = target_folder / f'{phase}_img'
    inst_fldr = target_folder / f'{phase}_inst'
    label_fldr = target_folder / f'{phase}_label'
    img_fldr.mkdir(exist_ok=False)
    inst_fldr.mkdir(exist_ok=False)
    label_fldr.mkdir(exist_ok=False)
    for ID in tqdm(IDs, desc=phase):
        shutil.move(f'all_img/{ID}.png', img_fldr / f'{ID}.png')
        shutil.move(f'all_label/{ID}_semantic.png', label_fldr / f'{ID}_semantic.png')
        shutil.move(f'all_inst/{ID}_instance.png', inst_fldr / f'{ID}_instance.png')

def main(target_folder):
    target_folder = Path(target_folder)
    target_folder.mkdir(exist_ok=False, parents=True)

    _handle(target_folder, 'train')
    _handle(target_folder, 'test')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('resize')
    parser.add_argument('--target', type=str, help='absolute path to target folder', default='GAN_dataset')
    args = parser.parse_args()
    print(args)
    print('Start preparing GAN dataset...')
    main(args.target)
    print('Preparing GAN dataset ended')
