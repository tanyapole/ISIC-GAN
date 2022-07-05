from ctypes import Array

import pandas as pd
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
import os
from numpy.random import RandomState


def load_mask(path: str) -> int:
    img = Image.open(path)
    data = np.asarray(img)
    return 1 if np.any(data == 255) else 0


def data_to_csv(real_prefix: str,
                generated_prefix: str,
                path: str,
                dataset_name: str,
                seeds,
                replace_first_percents,
                extend,
                items_to_use):
    masks = glob.glob(path + "*.png")
    if len(items_to_use) > 0:
        masks = list(filter(lambda x: any([True if i not in x else False for i in items_to_use]), masks))
    names = list(map(lambda x: x.split(path)[1], masks))
    items = list(map(lambda x: x.split("_"), names))
    codes = list(map(lambda x: x[1], items))
    types = list(map(lambda x: "_".join(x[3:]).split(".")[0], items))

    diseases = np.array(sorted(list(set(types))))
    assert len(types) == len(codes)
    assert len(types) == len(masks)
    result_dict = {}
    for path, code, typ in tqdm(list(zip(masks, codes, types))):
        if code in result_dict:
            labels = result_dict[code]
        else:
            labels = np.zeros(len(diseases))
            result_dict[code] = labels
        idx = np.where(diseases == typ)
        labels[idx] = load_mask(path)
    full = len(result_dict)
    part = int(full / 100 * replace_first_percents)
    for seed in seeds:
        rs = RandomState(seed)
        result = list(result_dict.items())
        result.sort(key=lambda x: x[0])
        result = rs.permutation(result)
        indices = list(map(lambda x: x[0], result))
        result = list(map(lambda x: x[1], result))
        if extend:
            indices_a = list(map(lambda x: os.path.join(generated_prefix, "ISIC_" + x + "_semantic_synthesized_image.jpg"), indices[:part]))
            indices_b = list(map(lambda x: os.path.join(real_prefix, "ISIC_" + x + ".jpg"),  indices[:part]))
            indices_c = list(map(lambda x: os.path.join(real_prefix, "ISIC_" + x + ".jpg"), indices[part:]))
            indices = indices_a + indices_b + indices_c
            result = result[:part] + result[:part] + result[part:]
        else:
            indices_a = list(map(lambda x: os.path.join(generated_prefix, "ISIC_" + x + "_semantic_synthesized_image.jpg"), indices[:part]))
            indices_b = list(map(lambda x: os.path.join(real_prefix, "ISIC_" + x + ".jpg"),  indices[part:]))
            indices = indices_a + indices_b

        frame = pd.DataFrame(result, index=indices, columns=diseases, dtype='int64')
        frame.to_csv(dataset_name.format(replace_first_percents, seed), index_label="images")


if __name__ == "__main__":
    with open("/Users/nduginets/PycharmProjects/master-diploma/splits/skin_lesion_test.txt", "r") as f:
        items_to_use = f.readlines()
        items_to_use = list(map(lambda x: x[:-5], items_to_use))
    ranges = [i for i in range(0, 10)]

    data_to_csv("images/ISIC2018_Task1-2_Training_Input",
                "images/pix2pix_result/pix2pix_result/label2skin/test_latest/images/",
                "/Users/nduginets/Desktop/images/ISIC2018_Task2_Training_GroundTruth_v3/", # ISIC2018_Task2_Validation_GroundTruth ISIC2018_Task2_Training_GroundTruth_v3
                "generated/train_1{}_{}.csv", ranges, 90, True, items_to_use)

    data_to_csv("images/ISIC2018_Task1-2_Training_Input",
                "images/pix2pix_result/pix2pix_result/label2skin/test_latest/images/",
                "/Users/nduginets/Desktop/images/ISIC2018_Task2_Training_GroundTruth_v3/", # ISIC2018_Task2_Validation_GroundTruth ISIC2018_Task2_Training_GroundTruth_v3
                "generated/train_1{}_{}.csv", ranges, 100, True, items_to_use)
