import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, sampler
from torchvision import models, datasets, transforms
from torchvision.utils import save_image

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


def a():
    m = nn.LogSoftmax()
    i = torch.randn(2)
    print(i)
    print("===")
    i1 = nn.Sigmoid()(i)
    print(i1)
    print("===")
    t = torch.Tensor([0, 0])
    c1 = nn.BCEWithLogitsLoss()
    c2 = nn.BCELoss()
    print("C1:", c1(i, t))
    print("C2:", c2(i1, t))


if __name__ == "__main__":
    """model = nn.Linear(20, 5)  # predict logits for 5 classes
    x = torch.randn(1, 20)
    y = torch.tensor([[1., 0., 1., 0., 0.]])  # get classA and classC as active

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-1)

    for epoch in range(20):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print('Loss: {:.3f}'.format(loss.item()))
    """
    writer.add_text("log", "heh")