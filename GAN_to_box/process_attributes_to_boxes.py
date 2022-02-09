import sys
import os

import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import os
from tqdm import tqdm
from PIL import Image
from joblib import Parallel, delayed

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


import queue


def bfs(image, mask, i, j, current_color, component_index):
    q = queue.Queue()
    q.put((i, j))
    while not q.empty():
        i, j = q.get()
        if i < 0 or j < 0:
            continue
        if i >= image.shape[0] or j >= image.shape[1]:
            continue
        if mask[i][j] == 0 and image[i][j] == current_color:
            mask[i][j] = component_index
            q.put((i - 1, j - 1))
            q.put((i - 1, j))
            q.put((i - 1, j + 1))

            q.put((i, j - 1))
            q.put((i, j))
            q.put((i, j + 1))

            q.put((i + 1, j - 1))
            q.put((i + 1, j))
            q.put((i + 1, j + 1))


def group_by_classes(image):
    component_color = 1
    mask = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if mask[i][j] == 0:
                color = image[i][j]
                bfs(image, mask, i, j, color, component_color)
                component_color += 1
    return image, mask


def create_rectangles(image, extend_to):
    img, msk = group_by_classes(image)

    group_classes = np.unique(msk)

    class_index = []
    i_size, j_size = img.shape

    for cls in group_classes:
        i_idx, j_idx = np.where(msk == cls)
        min_i_idx = i_idx.min()
        size_i_idx = i_idx.max() - min_i_idx

        min_j_idx = j_idx.min()
        size_j_idx = j_idx.max() - min_j_idx
        color = img[i_idx[0]][j_idx[0]]
        if color != 0 and size_i_idx != 0 and size_j_idx != 0:
            class_index.append((min_i_idx / i_size, min_j_idx / j_size, size_i_idx / i_size, size_j_idx / j_size))
    empty = (0, 0, 0, 0)
    res = [empty for _ in range(extend_to)]
    for i in range(min(extend_to, len(class_index))):
        res[i] = class_index[i]
    return res


# color != 0
"""
    for i in range(len(class_index)):
        for j in range(i + 1, len(class_index)):
            a_i_1, a_j_1 = class_index[i][2]
            a_i_2, a_j_2 = class_index[i][3]

            b_i_1, b_j_1 = class_index[j][2]
            b_i_2, b_j_2 = class_index[j][3]

            if a_i_1 < b_i_2 and a_i_2 > b_i_1 and a_j_1 < b_j_2 and a_j_2 > b_j_1:
                sq_a = (a_i_1 - a_i_2) ** 2 + (a_j_1 - a_j_2) ** 2
                sq_b = (b_i_1 - b_i_2) ** 2 + (b_j_1 - b_j_2) ** 2
                # case when rect_a in rect_b covered by square comparison
                if sq_a < sq_b:
                    tmp = class_index[i]
                    class_index[i] = class_index[j]
                    class_index[j] = tmp

    rectangle_image = np.zeros(msk.shape)
    for (cls, color, (min_i, min_j), (max_i, max_j)) in class_index:
        rectangle_image[min_i: max_i, min_j:max_j] = color
    return rectangle_image
"""

import GLOBAL as G
import csv


def create_csv_header(extend_to):
    header = ["name"]
    for i in range(extend_to):
        header.append("segm_x_{}".format(i))
        header.append("segm_y_{}".format(i))
        header.append("segm_x_size_{}".format(i))
        header.append("segm_y_size_{}".format(i))
    for d in G.ORDER:
        for i in range(extend_to):
            header.append("{}_x_{}".format(d, i))
            header.append("{}_y_{}".format(d, i))
            header.append("{}_x_size_{}".format(d, i))
            header.append("{}_y_size_{}".format(d, i))
    return header


def f_by_threshold(image):
    image[np.where(image < 100.0)] = 0
    image[np.where(image >= 100.0)] = 1
    return image


def process_single_image_pack(index, attrs, segments, extend_to):
    segment_file = segments[index]
    attribute_files = attrs[index: index + G.LABELS_SIZE]
    segment_borders = create_rectangles(f_by_threshold(imread(segment_file, flatten=True)), extend_to)
    attribute_borders = [create_rectangles(f_by_threshold(imread(i, flatten=True)), extend_to) for i in attribute_files]
    merged = [segment_borders] + attribute_borders

    identifier = segment_file.split("/")[-1][5:11]

    result = []
    for arr in merged:
        for items in arr:
            result.append(items[0])
            result.append(items[1])
            result.append(items[2])
            result.append(items[3])
    return identifier, result


def create_bounding_boxes(base_path, attribute_name, segmentation_name, export_to_file_name):
    attr_dir = os.path.join(base_path, attribute_name)
    segm_dir = os.path.join(base_path, segmentation_name)
    file_name = os.path.join(base_path, export_to_file_name)

    atr = sorted(glob.glob(os.path.join(attr_dir, '*.png')))
    sgts = sorted(glob.glob(os.path.join(segm_dir, "*.png")))
    extend_to = 15
    header = create_csv_header(extend_to)
    results = Parallel(n_jobs=8)(
        delayed(process_single_image_pack)(i, atr, sgts, extend_to) for i in tqdm(range(0, len(sgts))))
    with open(file_name, "w", encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in results:
            n, d = i
            writer.writerow([n] + d)


if __name__ == "__main__":
    """print(sys.argv)
    if len(sys.argv) != 5:
        print("expected base data path, got: ", sys.argv)
        exit(1)
    path = sys.argv[1]
    attr_dir = sys.argv[2]
    segm_dir = sys.argv[3]
    name = sys.argv[4]
    #create_bounding_boxes(path, attr_dir, segm_dir, name)
    """
    create_bounding_boxes("/Users/nduginets/PycharmProjects/master-diploma/GAN_to_box/test_data/0000150",
                           "attribute_512p", "seg_512p", "report.csv")
