import os
import os.path
import numpy as np
import pandas as pd
import torch.utils.data as data
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
import torch


class CSVDataset(data.Dataset):
    def __init__(self, root,
                 csv_file,
                 image_field,
                 loader=default_loader,
                 transform=None):
        self.root = root

        self.loader = loader
        self.image_field = image_field
        self.transform = transform
        self.data = pd.read_csv(csv_file, sep=None)
        self.target_fields = self.data.columns[1:]

    def __getitem__(self, index):
        path = os.path.join(self.root,
                            self.data.loc[index, self.image_field])

        sample = self.loader(path)
        sample = self.transform(sample)

        masks = []
        for t in np.array(self.data.loc[index, self.target_fields].array):
            img = transforms.ToTensor()(self.loader(os.path.join(self.root, t))).sum(dim=0)
            img = transforms.Resize((512, 1024))(img.view((1, img.shape[0], img.shape[1])))
            masks.append(img)

        masks = torch.cat(masks)
        int_masks = torch.tensor(masks, dtype=torch.int)
        return sample, masks, int_masks

    def __len__(self):
        return len(self.data)


class CSVDatasetWithName(CSVDataset):
    """
    CSVData that also returns image names.
    """

    def __init__(self, *args, **kwargs):
        super(CSVDatasetWithName, self).__init__(*args, **kwargs)
        # self.cache = {}

    def __getitem__(self, i):
        #if i in self.cache:
        #    return self.cache[i]
        name = self.data.loc[i, self.image_field]
        record = super().__getitem__(i), name
        #self.cache[i] = record
        return record


if __name__ == "__main__":
    dataset = CSVDatasetWithName(
        "/Users/nduginets/Desktop",
        "/Users/nduginets/PycharmProjects/master-diploma/segmentation_splits/validation.csv",
        'images', transform=transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 * 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Resize((512, 1024))
        ])
    )
    dataloader = DataLoader(dataset)
    for i in dataloader:
        print(i)
