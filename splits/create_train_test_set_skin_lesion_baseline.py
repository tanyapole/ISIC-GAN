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


def data_to_csv(prefix: str, path: str, dataset_name: str, seeds, items_to_use, only_items_to_use):
    masks = glob.glob(path + "*.png")
    if only_items_to_use:
        masks = list(filter(lambda x: any([True if i in x else False for i in items_to_use]), masks))
    else:
        masks = list(filter(lambda x: any([True if i not in x else False for i in items_to_use]), masks))
    names = list(map(lambda x: x.split(path)[1], masks))
    items = list(map(lambda x: x.split("_"), names))
    codes = list(map(lambda x: os.path.join(prefix, "ISIC_" + x[1] + ".jpg"), items))
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

    use_format = True if len(seeds) > 1 else False
    for seed in seeds:
        rs = RandomState(seed)
        result = list(result_dict.items())
        result.sort(key=lambda x: x[0])
        result = rs.permutation(result)
        indices = list(map(lambda x: x[0], result))
        result = list(map(lambda x: x[1], result))

        frame = pd.DataFrame(result, index=indices, columns=diseases, dtype='int64')
        if use_format:
            frame.to_csv(dataset_name.format(seed), index_label="images")
        else:
            frame.to_csv(dataset_name, index_label="images")


if __name__ == "__main__":
    with open("/Users/nduginets/PycharmProjects/master-diploma/splits/skin_lesion_test.txt", "r") as f:
        items_to_use = f.readlines()
        items_to_use = list(map(lambda x: x[:-5], items_to_use))

    data_to_csv("ISIC2018_Task1-2_Training_Input",
                "/Users/nduginets/Desktop/images/ISIC2018_Task2_Training_GroundTruth_v3/",
                "validation_skin_lesion.csv", [0], items_to_use, True)

    ranges = [i for i in range(0, 10)]
    data_to_csv("ISIC2018_Task1-2_Training_Input",
                "/Users/nduginets/Desktop/images/ISIC2018_Task2_Training_GroundTruth_v3/",
                "baseline_bussio/train_{}.csv", ranges, items_to_use, False)
