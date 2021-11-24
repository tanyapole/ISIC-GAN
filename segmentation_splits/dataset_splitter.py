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


def create_record(index, deseases, image_path, mask_path):
    return [os.path.join(image_path, "ISIC_" + index + ".jpg")] + [
        os.path.join(mask_path, "ISIC_" + index + "_attribute_" + d + ".png") for d in deseases]


def data_to_csv(masks_data_path: str,
                save_to_file_path: str,
                real_image_path: str,
                fake_image_path: str,
                masks_path: str,
                seeds,
                replace_first_percents,
                extend):
    masks = glob.glob(masks_data_path + "*.png")
    names = list(map(lambda x: x.split(masks_data_path)[1], masks))
    items = list(map(lambda x: x.split("_"), names))
    codes = list(map(lambda x: x[1], items))
    types = list(map(lambda x: "_".join(x[3:]).split(".")[0], items))

    diseases = np.array(sorted(list(set(types))))
    print(diseases)
    print(types)
    print(codes)
    assert len(types) == len(codes)
    assert len(types) == len(masks)
    codes = list(set(codes))
    full = len(codes)
    part = int(full / 100 * replace_first_percents)
    filled = [(c, diseases) for c in codes]
    for seed in seeds:
        rs = RandomState(seed)
        result = list(filled)
        result.sort(key=lambda x: x[0])
        result = rs.permutation(result)
        indexes = list(map(lambda x: x[0], result))
        indexes_a = indexes[:part]
        indexes_b = indexes[:part]
        indexes_c = indexes[part:]

        indexes_a = list(map(lambda i: create_record(i, diseases, fake_image_path, masks_path), indexes_a))
        indexes_b = list(map(lambda i: create_record(i, diseases, real_image_path, masks_path), indexes_b))
        indexes_c = list(map(lambda i: create_record(i, diseases, real_image_path, masks_path), indexes_c))

        if extend:
            result = indexes_a + indexes_b + indexes_c
        else:
            result = indexes_a + indexes_c
        frame = pd.DataFrame(result, columns=['images'] + list(diseases))
        frame.to_csv(save_to_file_path.format(replace_first_percents, seed), index_label="images", index=False)


if __name__ == "__main__":
    ranges = [i for i in range(0, 10)]

    """
    data_to_csv(
        "/Users/nduginets/Desktop/images/ISIC2018_Task2_Validation_GroundTruth/",
        "validation.csv",
        "images/ISIC2018_Task1-2_Validation_Input",
        None,
        "images/ISIC2018_Task2_Validation_GroundTruth/",
        ranges, 0, False)

    data_to_csv(
        "/Users/nduginets/Desktop/images/ISIC2018_Task2_Training_GroundTruth_v3/",
        "baseline/train_{}.csv",
        "images/ISIC2018_Task1-2_Training_Input",
        None,
        "images/ISIC2018_Task2_Training_GroundTruth_v3/",
        ranges, 0, False)
    """

    data_to_csv(
        "/Users/nduginets/Desktop/images/ISIC2018_Task2_Training_GroundTruth_v3/",
        "generated/train_{}_{}.csv",
        "images/ISIC2018_Task1-2_Training_Input",
        "pix2pix_result/label2skin/test_latest/images/",
        "images/ISIC2018_Task2_Training_GroundTruth_v3/",
        ranges, 20, False)

    data_to_csv(
        "/Users/nduginets/Desktop/images/ISIC2018_Task2_Training_GroundTruth_v3/",
        "generated/train_{}_{}.csv",
        "images/ISIC2018_Task1-2_Training_Input",
        "pix2pix_result/label2skin/test_latest/images/",
        "images/ISIC2018_Task2_Training_GroundTruth_v3/",
        ranges, 50, False)


    data_to_csv(
        "/Users/nduginets/Desktop/images/ISIC2018_Task2_Training_GroundTruth_v3/",
        "generated/train_{}_{}.csv",
        "images/ISIC2018_Task1-2_Training_Input",
        "pix2pix_result/label2skin/test_latest/images/",
        "images/ISIC2018_Task2_Training_GroundTruth_v3/",
        ranges, 80, False)


    data_to_csv(
        "/Users/nduginets/Desktop/images/ISIC2018_Task2_Training_GroundTruth_v3/",
        "generated/train_1{}_{}.csv",
        "images/ISIC2018_Task1-2_Training_Input",
        "pix2pix_result/label2skin/test_latest/images/",
        "images/ISIC2018_Task2_Training_GroundTruth_v3/",
        ranges, 20, True)

    data_to_csv(
        "/Users/nduginets/Desktop/images/ISIC2018_Task2_Training_GroundTruth_v3/",
        "generated/train_1{}_{}.csv",
        "images/ISIC2018_Task1-2_Training_Input",
        "pix2pix_result/label2skin/test_latest/images/",
        "images/ISIC2018_Task2_Training_GroundTruth_v3/",
        ranges, 50, True)


    data_to_csv(
        "/Users/nduginets/Desktop/images/ISIC2018_Task2_Training_GroundTruth_v3/",
        "generated/train_1{}_{}.csv",
        "images/ISIC2018_Task1-2_Training_Input",
        "pix2pix_result/label2skin/test_latest/images/",
        "images/ISIC2018_Task2_Training_GroundTruth_v3/",
        ranges, 80, True)
