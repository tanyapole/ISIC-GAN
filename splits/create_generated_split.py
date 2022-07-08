from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
import pandas as pd
from argparse import ArgumentParser

def main(data_root, generate_data_folder, ratios, seeds):
    data_root = Path(data_root)
    generate_data_folder = Path(generate_data_folder)
    data_parent_folder = data_root.parent
    original_images_folder = data_root / 'ISIC2018_Task1-2_Training_Input'
    original_attr_folder = data_root / 'ISIC2018_Task2_Training_GroundTruth_v3'
    train_test_split_folder = data_root / 'datasets' / 'skin'
    train_split_folder = train_test_split_folder / 'train_img'
    test_split_folder = train_test_split_folder / 'test_img'

    attrs = ['globules', 'milia_like_cyst', 'negative_network', 'pigment_network', 'streaks']

    def images_to_df(images, relative_to):
        rows = []
        for img in images:
            ID = '_'.join(img.name.split('.')[0].split('_')[:2])
            row = labels[ID]
            row['images'] = img.relative_to(relative_to)
            rows.append(row)
        return pd.DataFrame(columns=['images']+attrs, data=rows)

    # split
    train_ids = sorted(list(map(lambda p: p.name[:-len(p.suffix)], train_split_folder.iterdir())))
    test_ids = sorted(list(map(lambda p: p.name[:-len(p.suffix)], test_split_folder.iterdir())))
    assert len(set(train_ids).intersection(set(test_ids))) == 0, 'test leaking into train'

    # collect labels
    labels = {}
    for ID in tqdm(train_ids):
        labels[ID] = {}
        for attr in attrs:
            pt = original_attr_folder / f'{ID}_attribute_{attr}.png'
            img = Image.open(pt)
            labels[ID][attr] = int((np.array(img) > 0).any())

    for seed in seeds:

        # shuffle
        train_ids_2 = train_ids[:]
        train_ids_3 = train_ids[:]
        np.random.seed(seed)
        np.random.shuffle(train_ids_2)
        np.random.shuffle(train_ids_3)

        # original
        original_images = [original_images_folder / f'{ID}.jpg' for ID in train_ids_2]
        assert len(original_images) == len(train_ids), "selected wrong original images"
        orig_df = images_to_df(original_images, data_parent_folder)
        
        for ratio in ratios:

            # generated
            N1 = round(len(train_ids) * ratio)
            generated_images = [generate_data_folder / f'{ID}_semantic_synthesized_image.jpg' for ID in train_ids_3[:N1]]
            assert len(generated_images) == N1, "selected wrong number of generated images"
            for p in generated_images: 
                assert p.exists(), str(p)
            gen_df = images_to_df(generated_images, data_parent_folder)
            
            # save csv
            df = pd.concat([orig_df, gen_df], axis=0)
            target_pt = f'./generated_bussio/train_{int(ratio * 100) + 100}_{seed}.csv'
            df.to_csv(target_pt, index=False)

if __name__ == '__main__':
    parser = ArgumentParser('create split on generated data')
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--generated-data-folder', type=str, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--ratios', type=float, nargs='+', required=True)
    args = parser.parse_args()
    main(data_root=args.data_root, 
        generate_data_folder=args.generated_data_folder,
        ratios=args.ratios,
        seeds=args.seeds)