import sys
import os

import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import os
from tqdm import tqdm
from PIL import Image

import numpy

from numpy import (amin, amax, ravel, asarray, arange, ones, newaxis,
                   transpose, iscomplexobj, uint8, issubdtype, array)


def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    """
    if data.dtype == uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(uint8)


def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
    """Takes a numpy array and returns a PIL image.
    This function is only available if Python Imaging Library (PIL) is installed.
    The mode of the PIL image depends on the array shape and the `pal` and
    `mode` keywords.
    For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
    is given as 'F' or 'I' in which case a float and/or integer array is made.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Notes
    -----
    For 3-D arrays, the `channel_axis` argument tells which dimension of the
    array holds the channel data.
    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
    by default or 'YCbCr' if selected.
    The numpy array must be either 2 dimensional or 3 dimensional.
    """
    data = asarray(arr)
    if iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(numpy.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high, low=low,
                                 cmin=cmin, cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(asarray(pal, dtype=uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (arange(0, 256, 1, dtype=uint8)[:, newaxis] *
                       ones((3,), dtype=uint8)[newaxis, :])
                image.putpalette(asarray(pal, dtype=uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = amin(ravel(data))
        if cmax is None:
            cmax = amax(ravel(data))
        data = (data * 1.0 - cmin) * (high - low) / (cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(numpy.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(
                "wht something wrong https://github.com/scipy/scipy/blob/368dbad596a0bd0d5a88a7aec381fdc912440ee1/scipy/misc/pilutil.py#L286-L409")
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = numpy.flatnonzero(asarray(shape) == 3)[0]
        else:
            ca = numpy.flatnonzero(asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(
            "wht something wrong https://github.com/scipy/scipy/blob/368dbad596a0bd0d5a88a7aec381fdc912440ee1/scipy/misc/pilutil.py#L286-L409")

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image


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


def process_segmentation_map(base_path):
    atri_dir = os.path.join(base_path, 'attribute_resized/')
    mask_dir = os.path.join(base_path, 'segmentation_resized/')
    output_dir = os.path.join(base_path, 'semantic_map/')

    print(atri_dir)
    print(mask_dir)
    print(output_dir)

    file_name_arr = []  # [ISIC_00000, ISIC_000001, ISIC_000003, ...]
    for file in glob.glob(atri_dir + '*.png'):
        temp = file.split('/')[-1].split('_')
        file_name = temp[0] + '_' + temp[1]
        if file_name not in file_name_arr:
            file_name_arr.append(file_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for family in tqdm(file_name_arr):
        # Create a zero filled base image
        for i, file in enumerate(glob.glob(atri_dir + family + '*.png')):
            # Read the image
            read_image = imread(file, flatten=True)
            border_color = read_image[0, 0]
            read_image[read_image == border_color] = 0
            read_image[read_image > 0] = 255
            read_image = np.int8(read_image / 255)

            if i == 0:
                mask = imread(mask_dir + family + '_segmentation.png', flatten=True)
                base_image = np.ones(read_image.shape, dtype=int)  # Healthy Skin is 1
                border_mask_color = mask[0, 0]
                base_image[mask == border_mask_color] = 0
                mask[mask == border_mask_color] = 0
                mask[mask > 0] = 255
                mask = np.int8(mask / 255)
                base_image += mask  # Common Lesion is 2

            type_file = file.split('/')[-1].split('_')[3]

            if type_file == 'pigment':  # 3
                base_image += read_image
                if base_image[base_image > 3].any():
                    base_image[base_image > 3] = 3
            elif type_file == 'negative':  # 4
                base_image += read_image * 2
                if base_image[base_image > 4].any():
                    base_image[base_image > 4] = 4
            elif type_file.startswith('streaks'):  # 5
                base_image += read_image * 3
                if base_image[base_image > 5].any():
                    base_image[base_image > 5] = 5
            elif type_file == 'milia':  # 6
                base_image += read_image * 4
                if base_image[base_image > 6].any():
                    base_image[base_image > 6] = 6
            elif type_file.startswith('globules'):  # 7
                base_image += read_image * 5
                if base_image[base_image > 7].any():
                    base_image[base_image > 7] = 7
            else:
                print('ERROR: Invalid File Found!!!!')
        toimage(base_image, cmin=0, cmax=255).save(output_dir + family + '_semantic.png')


if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) != 2:
        print("expected base data path, got: ", sys.argv)
        exit(1)
    path = sys.argv[1]
    process_segmentation_map(path)
