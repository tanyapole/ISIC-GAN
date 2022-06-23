import os
import os.path
import numpy as np
import pandas as pd
import torch.utils.data as data
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms


class CSVDataset(data.Dataset):
    def __init__(self, root, csv_file, image_field,
                 loader=default_loader, transform=None,
                 target_transform=None, add_extension=None,
                 limit=None, random_subset_size=None,
                 split=None):
        self.root = root
        self.loader = loader
        self.image_field = image_field
        self.transform = transform
        self.target_transform = target_transform
        self.add_extension = add_extension

        self.data = pd.read_csv(csv_file, sep=None)
        self.target_fields = self.data.columns[1:]

        # Split
        if split is not None:
            with open(split, 'r') as f:
                selected_images = f.read().splitlines()
            self.data = self.data[self.data[image_field].isin(selected_images)]
            self.data = self.data.reset_index()

        if random_subset_size:
            self.data = self.data.sample(n=random_subset_size)
            self.data = self.data.reset_index()

        if type(limit) == int:
            limit = (0, limit)
        if type(limit) == tuple:
            self.data = self.data[limit[0]:limit[1]]
            self.data = self.data.reset_index()

        for target_field in self.target_fields:
            classes = list(self.data[target_field].unique())
            classes.sort()
            self.class_to_idx = {classes[i]: i for i in range(len(classes))}
            self.classes = classes

            print('{} found {} images from {} classes.'.format(target_field, len(self.data),
                                                               len(classes)))
            for class_name, idx in self.class_to_idx.items():
                n_images = dict(self.data[target_field].value_counts())
                print("    Class '{}' ({}): {} images.".format(
                    class_name, idx, n_images[class_name]))

    def __getitem__(self, index):
        path = os.path.join(self.root,
                            self.data.loc[index, self.image_field])
        if self.add_extension and ('.png' not in path) and ('.jpg' not in path):
            path = path + self.add_extension
        sample = self.loader(path)
        target = np.array(self.data.loc[index, self.target_fields].array, dtype='float32')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.data)


class CSVDatasetWithName(CSVDataset):
    """
    CSVData that also returns image names.
    """

    def __getitem__(self, i):
        """
        Returns:
            tuple(tuple(PIL image, int), str): a tuple
            containing another tuple with an image and
            the label, and a string representing the
            name of the image.
        """
        name = self.data.loc[i, self.image_field]
        return super().__getitem__(i), name


if __name__ == "__main__":
    dataset = CSVDatasetWithName(
        "/Users/nduginets/Desktop",
        "/Users/nduginets/PycharmProjects/master-diploma/splits/validation.csv",
        'images', transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(299, scale=(0.75, 1.0)),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 * 0.1),
            # transforms.ColorJitter(hue=0.2),
            transforms.ToTensor(),
            # transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )
    dataloader = DataLoader(dataset)
    for i in dataloader:
        print(i)
        break
