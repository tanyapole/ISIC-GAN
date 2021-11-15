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
                extend):
    masks = glob.glob(path + "*.png")
    names = list(map(lambda x: x.split(path)[1], masks))
    items = list(map(lambda x: x.split("_"), names))
    # codes = list(map(lambda x: os.path.join(prefix, "ISIC_" + x[1] + ".jpg"), items))
    codes = list(map(lambda x: x[1], items))
    types = list(map(lambda x: "_".join(x[3:]).split(".")[0], items))

    diseases = np.array(sorted(list(set(types))))
    print(diseases)
    print(types)
    print(codes)
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
    ranges = [i for i in range(0, 10)]
    """data_to_csv("images/ISIC2018_Task1-2_Training_Input",
                "pix2pix_result/label2skin/test_latest/images/",
                "/Users/nduginets/Desktop/images/ISIC2018_Task2_Training_GroundTruth_v3/", # ISIC2018_Task2_Validation_GroundTruth ISIC2018_Task2_Training_GroundTruth_v3
                "generated/train_1{}_{}.csv", ranges, 20, True)
    """

    data_to_csv("images/ISIC2018_Task1-2_Training_Input",
                "pix2pix_result/label2skin/test_latest/images/",
                "/Users/nduginets/Desktop/images/ISIC2018_Task2_Training_GroundTruth_v3/", # ISIC2018_Task2_Validation_GroundTruth ISIC2018_Task2_Training_GroundTruth_v3
                "generated/train_{}_{}.csv", ranges, 50, False)

    data_to_csv("images/ISIC2018_Task1-2_Training_Input",
                "pix2pix_result/label2skin/test_latest/images/",
                "/Users/nduginets/Desktop/images/ISIC2018_Task2_Training_GroundTruth_v3/", # ISIC2018_Task2_Validation_GroundTruth ISIC2018_Task2_Training_GroundTruth_v3
                "generated/train_{}_{}.csv", ranges, 80, False)
