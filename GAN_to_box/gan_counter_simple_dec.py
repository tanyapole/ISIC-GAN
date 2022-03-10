import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import VGG


class Generator(nn.Module):
    def __init__(self, c = 6, r = 16):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(100, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, c * r),
            nn.Unflatten(1, (1, c, r)),
            nn.Softmax(dim=3)
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, c = 6, r = 16):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * r, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


if __name__ == "__main__":
    t = torch.randn((2, 100))
    g = Generator()
    res = g(t)
    print(res)
    d = Discriminator()
    print(d(res))

    a = [0.0636, 0.0615, 0.0613, 0.0623, 0.0623, 0.0629, 0.0635, 0.0636,
           0.0629, 0.0616, 0.0618, 0.0645, 0.0610, 0.0619, 0.0631, 0.0622]
    print(sum(a))
