import sys
import os
# import the necessary packages
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from skimage import io
from scipy import misc
import numpy as np
import argparse
from PIL import Image
import glob
from tqdm import tqdm
import os
from joblib import Parallel, delayed


import numpy

from numpy import (amin, amax, ravel, asarray, arange, ones, newaxis,
                   transpose, iscomplexobj, uint8, issubdtype, array)


def fromimage(im, flatten=False, mode=None):
    if not Image.isImageType(im):
        raise TypeError("Input is not a PIL image.")

    if mode is not None:
        if mode != im.mode:
            im = im.convert(mode)
    elif im.mode == 'P':
        # Mode 'P' means there is an indexed "palette".  If we leave the mode
        # as 'P', then when we do `a = array(im)` below, `a` will be a 2-D
        # containing the indices into the palette, and not a 3-D array
        # containing the RGB or RGBA values.
        if 'transparency' in im.info:
            im = im.convert('RGBA')
        else:
            im = im.convert('RGB')

    if flatten:
        im = im.convert('F')
    elif im.mode == '1':
        # Workaround for crash in PIL. When im is 1-bit, the call array(im)
        # can cause a seg. fault, or generate garbage. See
        # https://github.com/scipy/scipy/issues/2138 and
        # https://github.com/python-pillow/Pillow/issues/350.
        #
        # This converts im from a 1-bit image to an 8-bit image.
        im = im.convert('L')

    a = array(im)
    return a


def imread(name, flatten=False, mode=None):
    im = Image.open(name)
    return fromimage(im, flatten=flatten, mode=mode)


def indian_code(base_path):
    # Directories
    atri_dir = os.path.join(base_path, 'attribute_512p/')
    image_dir = os.path.join(base_path, 'images_512p/')
    segmentation_dir = os.path.join(base_path, 'seg_512p/')
    output_dir = os.path.join(base_path, 'instance_map_no_border/')
    print("atri_dir", atri_dir)
    print("image_dir", image_dir)
    print("segmentation_dir", segmentation_dir)
    print("output_dir", output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_name_arr = []
    for file in glob.glob(atri_dir + '*.png'):
        temp = file.split('/')[-1].split('_')
        file_name = temp[0] + '_' + temp[1]
        if file_name not in file_name_arr:
            file_name_arr.append(file_name)

    def create_instance_map(family):
        # Create a zero filled base image
        # Load original image
        image = imread(image_dir + family + '.png')
        instance_map = np.zeros(image.shape[:2], dtype=int)
        segments = slic(img_as_float(image), n_segments=1000,
                        slic_zero=True, compactness=1, sigma=2)

        for i, file in enumerate(glob.glob(atri_dir + family + '*.png')):
            # Read Mask
            mask = imread(file)
            type_file = file.split('/')[-1].split('_')[3]
            if i == 0:
                segmentation = imread(segmentation_dir + family + '_segmentation.png', flatten=True)
                last_lesion = 2000
                last_background = 1000
                for v in np.unique(segments):
                    union = segmentation[segments == v]
                    if float(len(union[union > 0])) / float(len(union)) > 0.5:
                        instance_map[segments == v] = last_lesion
                        last_lesion += 1
                    else:
                        instance_map[segments == v] = last_background
                        last_background += 1

            if type_file == 'pigment':  # 3
                last = 3000
            elif type_file == 'negative':  # 4
                last = 4000
            elif type_file.startswith('streaks'):  # 5
                last = 5000
            elif type_file == 'milia':  # 6
                last = 6000
            elif type_file.startswith('globules'):  # 7
                last = 7000
            else:
                print('ERROR: Invalid File Found!!!!')
            # For each superpixel in the selected mask,
            # update the pixel values incrementing the value for each superpixel found.
            for v in np.unique(segments):
                union = mask[segments == v]
                if float(len(union[union > 0])) / float(len(union)) > 0.5:
                    instance_map[segments == v] = last
                    last += 1
        instance_map = instance_map.astype(np.uint32)
        im = Image.fromarray(instance_map)
        im.save(output_dir + family + '_instance.png')

    results = Parallel(n_jobs=8)(delayed(create_instance_map)(family) for family in tqdm(file_name_arr))
    print(results)


if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) != 2:
        print("expected base data path, got: ", sys.argv)
        exit(1)
    path = sys.argv[1]
    indian_code(path)
