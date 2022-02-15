from itertools import islice
import os
import numpy as np
import pandas as pd
import pretrainedmodels as ptm
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, sampler
from torchvision import models, datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
from dataset_loader import CSVDataset, DatasetMetadata
import params_loader as pl
import utils as utils
from sacred.observers import RunObserver
import logging
import math
import MY_GAN as GAN
import boundary_seeking_gan as BSGAN

np.set_printoptions(precision=4, suppress=True)
THRESHOLD = 0.5
lr = 0.0001
beta1 = 0.5
features_count = 360


class ModelWithSigmoidOut(nn.Module):

    def __init__(self, model):
        super(ModelWithSigmoidOut, self).__init__()
        self.model = model
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.model(input)
        x = self.sigmoid(x)
        return x


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, items_size):
        self.sum += val
        self.count += items_size
        self.avg = self.sum / self.count


class ModelOptimizerHolder(nn.Module):

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler):
        super(ModelOptimizerHolder, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, input):
        return self.model(input)

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    def step(self):
        self.optimizer.step()  # todo OR SCHEDULER???
        return self

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad()


def crate_thresholded_data(tensor):
    output_copy = torch.zeros_like(tensor)
    output_copy[tensor[:] <= THRESHOLD] = 0
    output_copy[tensor[:] > THRESHOLD] = 1
    return output_copy


def train_epoch(device,
                generator: ModelOptimizerHolder,
                discriminator: ModelOptimizerHolder,
                dataloaders,
                metric_holder,
                criterion,
                phase,
                epoch_number,
                metadata: DatasetMetadata):
    losses = {
        "fake_D": AverageMeter(),
        "true_D": AverageMeter(),
        "D": AverageMeter(),
        "true_G": AverageMeter(),
        "sum": AverageMeter()
    }
    accuracies = {
        "fake_D": AverageMeter(),
        "true_D": AverageMeter(),
        "D": AverageMeter(),
        "true_G": AverageMeter(),
        "sum": AverageMeter()
    }
    if phase == 'train':
        generator.train()
        discriminator.train()
        meters = metric_holder['train']
    else:
        generator.eval()
        discriminator.eval()
        meters = metric_holder['val']

    tqdm_loader = tqdm(dataloaders[phase])
    for data in tqdm_loader:
        generator.zero_grad()
        discriminator.zero_grad()

        bounding_boxes_data = Variable(data.to(device).type(torch.Tensor))
        batch_size = bounding_boxes_data.shape[0]

        valid = Variable(torch.empty(batch_size, 1, device=device).fill_(1.0), requires_grad=False)
        fake = Variable(torch.empty(batch_size, 1, device=device).fill_(0.0), requires_grad=False)

        generator.zero_grad()

        z = Variable(torch.tensor(np.random.normal(0, 1, (batch_size, 100)), dtype=torch.float, device=device))
        gen_boxes = generator(z)

        fake_g_output = discriminator(gen_boxes)
        g_loss = criterion(fake_g_output, valid)
        g_loss.backward()
        generator.step()

        discriminator.zero_grad()

        real_d_output = discriminator(bounding_boxes_data)
        real_loss = criterion(real_d_output, valid)
        fake_d_output = discriminator(gen_boxes.detach())
        fake_loss = criterion(fake_d_output, fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        discriminator.step()

        losses["fake_D"].update(fake_loss.sum().cpu().item(), batch_size)
        losses["true_D"].update(real_loss.sum().cpu().item(), batch_size)
        losses["D"].update((real_loss.sum() + fake_loss.sum()).cpu().item(), batch_size * 2)
        losses["true_G"].update(g_loss.sum().cpu().item(), batch_size)
        losses["sum"].update((g_loss + fake_loss + real_loss).sum().cpu().item(), batch_size * 3)

        fake_D_output_copy = crate_thresholded_data(fake_d_output)
        true_D_output_copy = crate_thresholded_data(real_d_output)
        output_copy = crate_thresholded_data(fake_g_output)

        accuracies["fake_D"].update(torch.sum(fake_D_output_copy == fake).item(), batch_size)
        accuracies["true_D"].update(torch.sum(true_D_output_copy == valid).item(), batch_size)
        accuracies["D"].update(
            (torch.sum(true_D_output_copy == valid) + torch.sum(fake_D_output_copy == fake)).item(),
            batch_size * 2)
        accuracies["true_G"].update(torch.sum(output_copy == valid).item(), batch_size)
        accuracies["sum"].update(
            torch.sum(fake_D_output_copy == fake).item() +
            torch.sum(true_D_output_copy == valid).item() +
            torch.sum(output_copy == valid).item(),
            batch_size * 3
        )

        """
        if phase == 'train':
            generator.zero_grad()
            discriminator.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            true_label = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)

            true_D_output = discriminator(bounding_boxes_data).view(-1)
            error_real_D = criterion(true_D_output, true_label)
            if phase == "train":
                error_real_D.backward()  # update gradient for Discriminator TRUE data
            noise = torch.randn(batch_size, 100, device=device)
            fake_G_output = generator(noise)  # .reshape(batch_size, features_count, 1, 1)

            fake_label = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)
            fake_D_output = discriminator(fake_G_output).view(-1)

            error_fake_D = criterion(fake_D_output, fake_label)

            if phase == 'train':
                discriminator.step()  # RECALCULATE WEIGHTS
                error_fake_D.backward()  # update gradient for Discriminator FALSE data
            error_D = error_real_D + error_fake_D  # COMMON ERROR

            ######################
            # GENERATOR
            ######################

            # true because generator think that's true
            # насколько хорошо генератор смог обмануть дискриминатор
            true_label = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
            generated_fake_data = generator(noise)  # .reshape(batch_size, features_count, 1, 1)
            output = discriminator(generated_fake_data).view(-1)
            error_G = criterion(output, true_label)
            if phase == 'train':
                error_G.backward()  # RECALCULATE WEIGHTS
                generator.step()

            losses["fake_D"].update(fake_D_output.sum().cpu().item(), batch_size)
            losses["true_D"].update(true_D_output.sum().cpu().item(), batch_size)
            losses["D"].update((true_D_output.sum() + fake_D_output.sum()).cpu().item(), batch_size * 2)
            losses["true_G"].update(output.sum().cpu().item(), batch_size)
            losses["sum"].update((true_D_output + fake_D_output + output).sum().cpu().item(), batch_size * 3)

            fake_D_output_copy = crate_thresholded_data(fake_D_output)
            true_D_output_copy = crate_thresholded_data(true_D_output)
            output_copy = crate_thresholded_data(output)

            accuracies["fake_D"].update(torch.sum(fake_D_output_copy == fake_label).item(), batch_size)
            accuracies["true_D"].update(torch.sum(true_D_output_copy == true_label).item(), batch_size)
            accuracies["D"].update(
                (torch.sum(true_D_output_copy == true_label) + torch.sum(fake_D_output_copy == fake_label)).item(),
                batch_size * 2)
            accuracies["true_G"].update(torch.sum(output_copy == true_label).item(), batch_size)
            accuracies["sum"].update(
                torch.sum(fake_D_output_copy == fake_label).item() +
                torch.sum(true_D_output_copy == true_label).item() +
                torch.sum(output_copy == true_label).item(),
                batch_size * 3
            )
        """
        tqdm_loader.set_postfix(loss=("D=" + str(losses["D"].avg), "G=" + str(losses["true_G"].avg)),
                                acc=("D=" + str(accuracies["D"].avg), "G=" + str(accuracies["true_G"].avg)),
                                _epoch=epoch_number)

    result_cell = {}
    result_cell['loss'] = {}
    for k, v in losses.items():
        result_cell['loss'][k] = v.avg
    result_cell['accuracy'] = {}
    for k, v in accuracies.items():
        result_cell['accuracy'][k] = v.avg
    meters.add_record(epoch_number, result_cell)


def main(train_csv,
         val_csv,
         epochs: int,
         batch_size: int,
         num_workers,
         experiment_path,
         model_name):
    last_model_G_path = os.path.join(experiment_path, "last_model_G.pth")
    last_model_D_path = os.path.join(experiment_path, "last_model_D.pth")
    train_metrics = utils.Dumper(os.path.join(experiment_path, "train_metrics.json"))
    # test_metrics = utils.Dumper(os.path.join(experiment_path, "test_metrics.json"))
    metric_holder = {
        'train': train_metrics,
        # 'val': test_metrics
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "my":
        modelG = GAN.Generator()  # ModelWithSigmoidOut(GAN.Generator())
        modelD = GAN.Discriminator()  # ModelWithSigmoidOut(GAN.Discriminator())
    elif model_name == "boundary-seeking-gan":
        modelG = BSGAN.Generator()  # ModelWithSigmoidOut(GAN.Generator())
        modelD = BSGAN.Discriminator()  # ModelWithSigmoidOut(GAN.Discriminator())
    modelG.to(device)
    modelD.to(device)

    latest_known_epoch = train_metrics.latest_key(-1)
    logging.info("detected epoch number: {} of: {}".format(latest_known_epoch, epochs))
    if latest_known_epoch == -1:
        epochs_list = [i for i in range(epochs)]
        logging.info("start from begining")
    else:
        epochs_list = [i for i in range(latest_known_epoch + 1, epochs)]
        logging.info("start from epoch number: {}".format(latest_known_epoch + 1))
        modelG = torch.load(last_model_G_path)
        modelD = torch.load(last_model_D_path)

    # image_name, target
    train_ds = CSVDataset(train_csv)
    # val_ds = CSVDataset(val_csv)

    dataset_metadata = train_ds.metadata

    dl_train = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)
    # dl_val = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    dataloaders = {
        'train': dl_train,
        # 'val': dl_val,
    }

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    optimizerG = optim.Adam(modelG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(modelD.parameters(), lr=lr, betas=(beta1, 0.999))

    schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG,
                                                milestones=[25],
                                                gamma=0.1)

    schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD,
                                                milestones=[25],
                                                gamma=0.1)

    gen = ModelOptimizerHolder(modelG, optimizerG, schedulerG)
    discr = ModelOptimizerHolder(modelD, optimizerD, schedulerD)

    for epoch in epochs_list:
        logging.debug('train epoch {}/{}'.format(epoch + 1, epochs))
        train_epoch(device, gen, discr, dataloaders, metric_holder, criterion, 'train', epoch, dataset_metadata)

        # logging.debug('val epoch {}/{}'.format(epoch + 1, epochs))
        # train_epoch(device, gen, discr, dataloaders, metric_holder, criterion, 'val', epoch, dataset_metadata)
        # logging.debug('-' * 40)

        # gen.step()
        # discr.step()
        torch.save(gen.model, last_model_G_path)
        torch.save(discr.model, last_model_D_path)


if __name__ == "__main__":
    if os.path.exists("/Users/nduginets/Desktop"):
        path = "/Users/nduginets/PycharmProjects/master-diploma/GAN_to_box/test_data/0000150/test_report.csv"
        params = pl.initialize([
            '--train_csv', path,
            "--validate_csv", path,
            "--epochs", "400",
            "--result_dir", "/Users/nduginets/Desktop",
            "--experiment_name", "tmp",
            "--num_workers", "0",  # stupid Mac os!!!!
            "--batch_size", "7",
            "--model_name", "boundary-seeking-gan",
        ])
    else:
        params = pl.initialize()

    ex_path = os.path.join(params.result_dir, params.experiment_name)
    main(
        params.train_csv,
        params.validate_csv,
        params.epochs,
        params.batch_size,
        params.num_workers,
        ex_path,
        params.model_name)
