import pandas as pd
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader


class DatasetMetadata:

    def __init__(self, target_fields):
        self.target_fields = target_fields
        map_by_name = list(map(lambda x: x.split("_")[0], self.target_fields))
        self.unique_names = set(list(map(lambda x: x.split("_")[0], self.target_fields)))
        self.offset_map = {n: (map_by_name.index(n), len(map_by_name) - map_by_name[::-1].index(n) - 1) for n in
                           self.unique_names}


class CSVDataset(data.Dataset):
    def __init__(self, csv_file, use_augumentation=True):
        self.pandas_data = pd.read_csv(csv_file, sep=None)
        target_fields = self.pandas_data.columns[1:]
        self.data = [torch.tensor(self.pandas_data.loc[i, self.pandas_data.columns[1:]].values, dtype=torch.float)
                     # .reshape(360, 1, 1) todo removed
                     for i in range(len(self.pandas_data))
                     ]
        self.metadata = DatasetMetadata(target_fields)
        self.use_augumentation = use_augumentation

    def __getitem__(self, index):
        if self.use_augumentation:
            t = torch.randint(95, 105, (360,)) / 100.0
            return self.data[index] * t
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    path = "/Users/nduginets/PycharmProjects/master-diploma/GAN_to_box/test_data/0000150/test_report.csv"
    dataset = CSVDataset(path)
    dataloader = DataLoader(dataset)
    for i in dataloader:
        print(i)
        print(i.shape)
        break
