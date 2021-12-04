import pandas as pd
import numpy as np


def prt(data):
    cnt = len(data)

    print("streaks", sum(data['streaks']) / cnt)
    print("milia_like_cyst", sum(data['milia_like_cyst']) / cnt)
    print("negative_network", sum(data['negative_network']) / cnt)
    print("pigment_network", sum(data['pigment_network']) / cnt)
    print("globules", sum(data['globules']) / cnt)

    print("===" * 20)

    print("streaks", sum(data['streaks']))
    print("milia_like_cyst", sum(data['milia_like_cyst']))
    print("negative_network", sum(data['negative_network']))
    print("pigment_network", sum(data['pigment_network']))
    print("globules", sum(data['globules']))


def prt_a():
    prt(data=pd.read_csv("/Users/nduginets/PycharmProjects/master-diploma/splits/validation.csv"))
    print("===" * 50)
    print("===" * 50)
    print("===" * 50)
    prt(data=pd.read_csv("/Users/nduginets/PycharmProjects/master-diploma/splits/baseline/train_1.csv"))


def print_distribution():
    with open("/Users/nduginets/PycharmProjects/master-diploma/splits/skin_lesion_test.txt", "r") as f:
        names = f.readlines()
        names = list(map(lambda x: "images/ISIC2018_Task1-2_Training_Input/" + x[:-1], names))
        print(names)
    data = pd.read_csv("/Users/nduginets/PycharmProjects/master-diploma/splits/baseline/train_1.csv")
    res = []
    for n in names:
        res.append(data[data['images'] == n])
    data = pd.concat(res)
    prt(data)

if __name__ == "__main__":
    print_distribution()

    print("===" * 50)
    print("===" * 50)
    print("===" * 50)

    prt_a()