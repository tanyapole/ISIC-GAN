import pandas as pd
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np

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
    def __init__(self, csv_file):
        self.pandas_data = pd.read_csv(csv_file, sep=None)
        target_fields = self.pandas_data.columns[1:]
        self.data = [torch.tensor(self.pandas_data.loc[i, self.pandas_data.columns[1:]].values, dtype=torch.float)
                     for i in range(len(self.pandas_data))]
        self.count_data = [CSVDataset.fill_cnt_tensor(i, False) for i in self.data]
        self.count_data_augumented = [CSVDataset.fill_cnt_tensor(i, True) for i in self.data]

        self.metadata = DatasetMetadata(target_fields)


    def __getitem__(self, index):
        t = torch.randint(95, 105, (360,)) / 100.0
        return (self.data[index], self.data[index] * t), (self.count_data[index], self.count_data_augumented[index])

    def __len__(self):
        return len(self.data)

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


if __name__ == "__main__":
    print(np.random.uniform(0.00001, 10**(-20)))
    print(np.random.uniform(0.00001, 10 ** (-20)))
    print(np.random.uniform(0.00001, 10 ** (-20)))
    path = "/Users/nduginets/PycharmProjects/master-diploma/GAN_to_box/test_data/isic_2018_boxes_shifted.csv"
    dataset = CSVDataset(path)
    dataloader = DataLoader(dataset)
    for i in dataloader:
        print(i[1])
        break
