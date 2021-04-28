import glob
import numpy as np
import shutil
import sys
import os

if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) != 2:
        print("expected base data path, got: ", sys.argv)
        exit(1)
    path = sys.argv[1]
    lesion_arr = []
    np.random.seed(0)

    for lesion in glob.glob(os.path.join(path, 'datasets/skin/train_label/*.png')):
        print(lesion)
        case = lesion.split('/')[-1].split('_')[1]
        print(case)
        lesion_arr.append(case)
    print(lesion_arr)

    # Randomly select 150 samples to compose our test dataset.
    test = np.random.choice(lesion_arr, 250, replace=False)
    # Move the selected files to the correspondent test directory.
    for case in test:
        shutil.move(os.path.join(path, 'datasets/skin/train_label/ISIC_' + case + '_semantic.png'),
                    os.path.join(path, 'datasets/skin/test_label/'))

        shutil.move(os.path.join(path, 'datasets/skin/train_inst/ISIC_' + case + '_instance.png'),
                    os.path.join(path, 'datasets/skin/test_inst/'))

        shutil.move(os.path.join(path, 'datasets/skin/train_img/ISIC_' + case + '.png'),
                    os.path.join(path, 'datasets/skin/test_img/'))
