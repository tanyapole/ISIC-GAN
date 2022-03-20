import pandas as pd
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict

import random as rnd

rnd.seed(1488)


class DatasetMetadata:

    def __init__(self, target_fields):
        self.target_fields = target_fields
        map_by_name = list(map(lambda x: x.split("_")[0], self.target_fields))
        self.unique_names = set(list(map(lambda x: x.split("_")[0], self.target_fields)))
        self.offset_map = {n: (map_by_name.index(n), len(map_by_name) - map_by_name[::-1].index(n) - 1) for n in
                           self.unique_names}
        self.count = len(map_by_name)

        self.torch_vector = torch.zeros(self.count, dtype=torch.float)
        for i in range(self.count // 4):
            p = i * 4
            self.torch_vector[p + 2] = 2
            self.torch_vector[p + 3] = 2


class CSVDataset(data.Dataset):
    def __init__(self, csv_file, first_items=100):
        self.pandas_data = pd.read_csv(csv_file, sep=None)
        target_fields = self.pandas_data.columns[1:]
        self.data = [torch.tensor(self.pandas_data.loc[i, self.pandas_data.columns[1:]].values, dtype=torch.float)
                     for i in range(len(self.pandas_data))]
        self.count_data = [CSVDataset.fill_cnt_tensor(i, False) for i in self.data]

        self.same_size_count_data = CSVDataset.extend_dataset_same_size(self.count_data, first_items)
        self.metadata = DatasetMetadata(target_fields)

    def __getitem__(self, index):

        return (0, 0), (self.same_size_count_data[index], 0)

    def __len__(self):
        return len(self.same_size_count_data)

    @staticmethod
    def fill_cnt_tensor(t, augument):
        zeros = torch.zeros((6, 16))
        for des in range(0, 6):
            cnt = 0
            for i in range(0, 15):
                offset = des * (15 * 4) + i * 4
                if t[offset] != -1:
                    cnt += 1
            # if cnt != 0:
            zeros[des][cnt] = 1
        if augument:
            for i in range(0, 6):
                for j in range(0, 16):
                    if zeros[i][j] == 0:
                        zeros[i][j] = np.random.uniform(0.00001, 10 ** (-20))
                    else:
                        zeros[i][j] = 1 - np.random.uniform(0.00001, 10 ** (-20))
        return zeros

    @staticmethod
    def calculate_vector_cnt(data):
        res = []
        for v in data:
            res.append(tuple(torch.argmax(v, dim=1).numpy()))

        cnt_dict = defaultdict(int)
        for v in res:
            cnt_dict[v] += 1
        return cnt_dict

    @staticmethod
    def extend_dataset_same_size(data, first_items=100):
        map_cnt = CSVDataset.calculate_vector_cnt(data)
        max_size = max(map_cnt.values())
        new_data_set = []

        for idx, (tensor_tuple, _) in enumerate(sorted([k for k in map_cnt.items()], key=lambda x: x[1], reverse=True)):
            tensor_list = [CSVDataset.create_tensor(tensor_tuple)] * max_size
            new_data_set.extend(tensor_list)
            if idx == first_items:
                break

        rnd.shuffle(new_data_set)
        return new_data_set

    @staticmethod
    def create_tensor(tuple_data: tuple):
        zeros = torch.zeros((6, 16), dtype=torch.float)
        for idx, pos in enumerate(tuple_data):
            zeros[idx][pos] = 1.0
        return zeros


if __name__ == "__main__":
    print(np.random.uniform(0.00001, 10 ** (-20)))
    print(np.random.uniform(0.00001, 10 ** (-20)))
    print(np.random.uniform(0.00001, 10 ** (-20)))
    path = "/Users/nduginets/PycharmProjects/master-diploma/GAN_to_box/test_data/isic_2018_boxes_shifted.csv"
    dataset = CSVDataset(path)
    dataloader = DataLoader(dataset)
    for i in dataloader:
        print(i[1])
        break
