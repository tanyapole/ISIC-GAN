import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, sampler
from torchvision import models, datasets, transforms
from torchvision.utils import save_image
import pandas as pd

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


if __name__ == "__main__":
    train_data = pd.read_csv(train_csv)
    train_data_cnt = len(train_data)
    inverse_labels_distribution = torch.tensor([1 - (sum(train_data[l]) / train_data_cnt) for l in labels]).to(device)