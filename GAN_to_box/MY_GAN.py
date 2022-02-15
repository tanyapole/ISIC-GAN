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

# see attribute_512p_box use dataset without box
# see seg_512p_box use dataset without box
# see images_512p

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64


class Generator(nn.Module):

    def __init__(self, output_size=360, dropout=0.5):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 1024),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(2048, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, output_size),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, output_size=360, dropout=0.5):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(output_size, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Dropout(p=dropout),

            nn.Linear(2048, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),

            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


"""
class Generator(nn.Module):

    def __init__(self, output_size=360, dropout=0.5):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),


            nn.Conv2d(3, output_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(output_size),
            nn.ReLU(True),

            nn.Conv2d(output_size, output_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(output_size),
            nn.ReLU(True),

            nn.Conv2d(output_size, output_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(output_size),
            nn.ReLU(True),

            nn.Conv2d(output_size, output_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(output_size),
            nn.ReLU(True),

            nn.Conv2d(output_size, output_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(output_size),
            nn.ReLU(True),

            nn.Conv2d(output_size, output_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(output_size),
            nn.ReLU(True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(output_size, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, output_size),
        )

    def forward(self, input):
        input = self.main(input)
        input = torch.flatten(input, 1)
        return self.classifier(input)


class Discriminator(nn.Module):
    def __init__(self, output_size=360, dropout=0.5):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            #
            nn.ConvTranspose2d(output_size, 3, 64, 1, 0, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        return self.main(input)
"""

if __name__ == "__main__":
    t = torch.randn((10, 100))
    g = Generator()
    res = g(t)
    d = Discriminator()

    print(d(res))
