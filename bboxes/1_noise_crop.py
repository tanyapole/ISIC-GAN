# %%

import numpy as np
import matplotlib.pyplot as plt
import math
import glob
import json
import os
import pandas as pd
import copy
from scipy.ndimage.interpolation import shift
from tqdm import tqdm

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from PIL import Image


def main(base_path):
    base_path = "/nfs/home/nduginets/"
    dataset_path = os.path.join(base_path, "master-diploma/bboxes/bounding_boxes_metadata.csv")

    frame = pd.read_csv(dataset_path)

    imgs_path = "/mnt/tank/scratch/nduginets/images/pix2pix_datasets/base_dataset_folder/images_512p"

    frame = pd.read_csv(dataset_path)
    images_list = sorted(glob.glob(os.path.join(imgs_path, "*.png")))

    images_names = list(map(lambda x: x.split("/")[-1].split(".")[0], images_list))

    # %%

    IMAGE_PATH = "/mnt/tank/scratch/nduginets/images/noise_bboxes_v2/images"
    SEGMENTATION_PATH = "/mnt/tank/scratch/nduginets/images/noise_bboxes_v2/segmentation"
    ATTRIBUTES_PATH = "/mnt/tank/scratch/nduginets/images/noise_bboxes_v2/attributes"

    os.makedirs(IMAGE_PATH, exist_ok=True)
    os.makedirs(SEGMENTATION_PATH, exist_ok=True)
    os.makedirs(ATTRIBUTES_PATH, exist_ok=True)

    type_path_offset_list = [
        ("image", IMAGE_PATH),
        ("segmentation", SEGMENTATION_PATH),
        ("attribute_globules", ATTRIBUTES_PATH),
        ("attribute_milia_like_cyst", ATTRIBUTES_PATH),
        ("attribute_negative_network", ATTRIBUTES_PATH),
        ("attribute_pigment_network", ATTRIBUTES_PATH),
        ("attribute_streaks", ATTRIBUTES_PATH),
    ]

    multipliers = [1, 1.1, 1.2, 1.56]
    adds = [-0.2, -0.15, -0.1, 0, 0.1, 0.15, 0.2]

    # %%

    #  4 * 15
    sz = 4 * 15
    columns = frame.columns[1:]

    attributes = [
        columns[i * sz: (i + 1) * sz]
        for i in range(6)
    ]

    # %%

    def fill_cnt_tensor(t):
        zeros = np.zeros((6, 16))
        for des in range(0, 6):
            cnt = 0
            for i in range(0, 15):
                offset = des * sz + i * 4
                if t[offset] != 0:
                    cnt += 1
            zeros[des][cnt] = 1
        return tuple(np.argmax(zeros, axis=1))

    fill_cnt_tensor(frame.iloc[2][1:])

    map_tuple_to_indexes = {}

    for idx in range(len(frame.index)):
        image_coordinates = frame.iloc[idx][1:]
        tpl = fill_cnt_tensor(image_coordinates)
        if tpl not in map_tuple_to_indexes:
            map_tuple_to_indexes[tpl] = (0, [])
        new_cnt = map_tuple_to_indexes[tpl][0] + 1
        new_lsr = map_tuple_to_indexes[tpl][1]
        new_lsr.append(idx)
        map_tuple_to_indexes[tpl] = (new_cnt, new_lsr)

    list_tuple_to_indexes = sorted([(k, v[0], v[1]) for k, v in map_tuple_to_indexes.items()], key=lambda x: x[1],
                                   reverse=True)

    # %%

    lbls = ["segm", "globules", "milia_like_cyst", "negative_network", "pigment_network", "streaks"]

    def populate(lbl_list, max_cnt=100):
        result = []
        selected_tuples = set()
        for l in lbl_list:
            cnt = 0
            for r, _, items in filter(lambda x: x[0][lbls.index(l)] > 0, list_tuple_to_indexes):
                if cnt == max_cnt:
                    continue
                if r in selected_tuples:
                    continue
                cnt += 1
                selected_tuples.add(r)
                result.append((r, items[0]))
        return result

    tuple_index_to_noise = populate(["streaks", "negative_network", "globules", "milia_like_cyst", "negative_network"],
                                    25)

    # %%

    def fill_zeros(index, number_to_fill='9'):
        number = str(index)
        return "ISIC_" + number_to_fill * (7 - len(number)) + number

    print(fill_zeros(1_00_00_00))

    # %%

    print(tuple_index_to_noise[0][1])

    frame.iloc[152]

    # %%

    def noise_row(row, x_offset, y_offset, x_wide, y_wide):
        for j, attribute in enumerate(attributes):
            for i, a in enumerate(attribute):
                if "x_size" in a:
                    row[j * sz + i] *= x_wide
                elif "y_size" in a:
                    row[j * sz + i] *= y_wide
                elif "x" in a:
                    row[j * sz + i] += x_offset
                elif "y" in a:
                    row[j * sz + i] += y_offset
                else:
                    raise Exception("!!!!")
        return row

    def draw_image(row, offset, shape):
        image = np.zeros((shape[0], shape[1]), dtype=np.int32)
        calibrate_x = lambda s: max(0, min(s, shape[0]))
        calibrate_y = lambda s: max(0, min(s, shape[1]))

        parts = row[offset * sz: (offset + 1) * sz]
        for idx in range(0, len(parts), 4):
            x = calibrate_x(int(parts[idx] * shape[0]))
            y = calibrate_y(int(parts[idx + 1] * shape[1]))
            x_sz = calibrate_x(int(parts[idx + 2] * shape[0]))
            y_sz = calibrate_y(int(parts[idx + 3] * shape[1]))

            image[x: x + x_sz, y: y + y_sz] = 1
        return image

    # %%

    from skimage import data, color
    from skimage.transform import rescale, resize, downscale_local_mean

    def fill_with_mean_image_part(image_to_populate, base_image, row, offset):
        shape = base_image.shape
        mask = draw_image(row, offset, shape)
        mean = np.mean(base_image, axis=(0, 1))
        image_to_populate[mask == 1] = mean
        return image_to_populate

    def move_image_part(image_to_populate, base_image, row, offset, shift_value, rescale_value):
        shape = base_image.shape
        mask = draw_image(row, offset, shape)

        calibrate_x = lambda s: int(max(0, min(s, shape[0])))
        calibrate_y = lambda s: int(max(0, min(s, shape[1])))

        for pos in range(sz // 4):
            x_o = calibrate_x(row[offset * sz + pos * 4] * shape[0])
            y_o = calibrate_y(row[offset * sz + pos * 4 + 1] * shape[1])
            x_s = calibrate_x(row[offset * sz + pos * 4 + 2] * shape[0])
            y_s = calibrate_y(row[offset * sz + pos * 4 + 3] * shape[1])
            subpart = base_image[x_o:x_o + x_s, y_o:y_o + y_s]
            if subpart.shape[0] == 0 or subpart.shape[1] == 0:
                continue
            # print(subpart.shape)
            subpart_rescaled = rescale(subpart, (rescale_value, rescale_value, 1))

            x_s_n = calibrate_x(x_o + shape[0] * shift_value)
            y_s_n = calibrate_y(y_o + shape[1] * shift_value)
            x_e_n = calibrate_x(x_s_n + calibrate_x(x_s * rescale_value))
            y_e_n = calibrate_y(y_s_n + calibrate_y(y_s * rescale_value))

            x_len = x_e_n - x_s_n
            y_len = y_e_n - y_s_n

            image_to_populate[x_s_n: x_e_n, y_s_n:y_e_n] = subpart_rescaled[:x_len, :y_len]

        return image_to_populate

    def make_new_image(img, row, shift_value, rescale_value):
        image_to_populate = np.copy(img)
        for i in range(0, 6):
            image_to_populate = fill_with_mean_image_part(image_to_populate, img, row, i)

        for i in range(0, 6):
            image_to_populate = move_image_part(image_to_populate, img, row, i, shift_value, rescale_value)
        return image_to_populate

    def make_row_from_images(row, index, shift_value, rescale_value):
        name = os.path.join(imgs_path, row['name'] + ".png")
        row = np.array(row[1:])
        images = []
        img = resize(np.array(Image.open(name)), (512, 512, 3))
        shifted_image = make_new_image(img, row, shift_value, rescale_value)
        images.append(shifted_image)
        noised_row = noise_row(row, shift_value, shift_value, rescale_value, rescale_value)
        for i in range(6):
            mask = draw_image(noised_row, i, img.shape)
            images.append(mask)
        assert len(images) == 7

        result_files = []
        for i in range(0, len(images)):
            type_name, path = type_path_offset_list[i]
            full_path = os.path.join(path, fill_zeros(index) + "_" + type_name + ".png")
            Image.fromarray((images[i] * 255).astype(np.uint8)).save(full_path)
            result_files.append(full_path)
        return result_files

    res = make_row_from_images(frame.iloc[68], 1, 0, 1)

    print(np.unique(np.array(Image.open(res[0]))))
    print(np.unique(np.array(Image.open(res[1]))))
    print(np.unique(np.array(Image.open(res[2]))))
    print(np.unique(np.array(Image.open(res[3]))))
    print(np.unique(np.array(Image.open(res[4]))))
    print(np.unique(np.array(Image.open(res[5]))))

    # %%

    cnt = 10
    fig, plots = plt.subplots(nrows=cnt, ncols=8, figsize=(10, 30))

    for idx, (_, n) in enumerate(tuple_index_to_noise[0:cnt]):
        row = np.array(frame.iloc[n][1:])
        name = os.path.join(imgs_path, frame.iloc[n]['name'] + ".png")
        images = []
        img = resize(np.array(Image.open(name)), (512, 512, 3))

        images.append(img)

        x, y = (-0.3, 1.2)
        images.append(make_new_image(img, row, x, y))
        nosied_row = noise_row(row, x, x, y, y)
        for i in range(6):
            mask = draw_image(nosied_row, i, img.shape)
            images.append(mask)

        for jdx, img in enumerate(images):
            plots[idx][jdx].imshow(img)
    plt.show()

    # %%

    print(tuple_index_to_noise[0])

    list_of_a_b_idx = []
    for (_, p) in tuple_index_to_noise:
        for a in adds:
            for b in multipliers:
                list_of_a_b_idx.append((a, b, p))
    len(list_of_a_b_idx)

    # %%

    from joblib import Parallel, delayed

    def print_some(idx, v):
        row = frame.iloc[v[2]]
        a = v[0]
        b = v[1]
        make_row_from_images(row, idx, a, b)

    res = Parallel(n_jobs=30)(delayed(print_some)(idx, v) for idx, v in tqdm(list(enumerate(list_of_a_b_idx))))
