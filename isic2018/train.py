from itertools import islice
import os
import numpy as np
import pandas as pd
import pretrainedmodels as ptm
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, sampler
from torchvision import models, datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
from dataset_loader import CSVDatasetWithName
import params_loader as pl
import utils as utils
from sacred.observers import RunObserver
import logging
import math

np.set_printoptions(precision=4, suppress=True)
THRESHOLD = 0.5


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
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, items_size):
        self.val = val
        self.sum += val
        self.count += items_size
        self.avg = self.sum / self.count


# todo    '/Users/nduginets/Desktop' '/mnt/tank/scratch/nduginets/results-comet-gans'


def train_epoch(device, model, dataloaders, metric_holder, criterion, optimizer, phase,
                epoch_number, total_epoch_count,
                label_names,
                batches_per_epoch=None):
    losses = AverageMeter()
    accuracies = AverageMeter()
    predicted_by_classes = {i: [] for i in label_names}
    labels_by_classes = {i: [] for i in label_names}
    result_cell = {i: {} for i in label_names}
    if phase == 'train':
        model.train()
        meters = metric_holder['train']
    else:
        model.eval()
        meters = metric_holder['val']

    if batches_per_epoch:
        tqdm_loader = tqdm(
            islice(dataloaders['train'], 0, batches_per_epoch),
            total=batches_per_epoch)
    else:
        tqdm_loader = tqdm(dataloaders[phase], initial=epoch_number, total=total_epoch_count)
    for data in tqdm_loader:
        (inputs, labels), name = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        if phase == 'train':
            optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)

            output_copy = torch.zeros_like(outputs)
            output_copy[outputs[:, :] <= THRESHOLD] = 0
            output_copy[outputs[:, :] > THRESHOLD] = 1

            loss = criterion(outputs, labels)

            if phase == 'train':
                loss.backward()
                optimizer.step()

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(torch.sum(output_copy == labels).item(), (output_copy.shape[0] * output_copy.shape[1]))

        for idx, label_name in enumerate(label_names):
            labels_by_classes[label_name] += list(labels.cpu().data.numpy()[:, idx])
            predicted_by_classes[label_name] += list(output_copy.cpu().data.numpy()[:, idx])
        tqdm_loader.set_postfix(loss=losses.avg, acc=accuracies.avg)

    for idx, label_name in enumerate(label_names):
        real = np.array(labels_by_classes[label_name])
        predicted = np.array(predicted_by_classes[label_name])
        try:
            score = roc_auc_score(real, predicted)
            result_cell[label_name]['auc'] = score
        except ValueError as e:
            logging.error("Error while calc roc auc: %s" % e)
            result_cell[label_name]['auc'] = math.nan
        cm = confusion_matrix(real, predicted)
        result_cell[label_name]['cm'] = cm.tolist()
        result_cell[label_name]['f1_binary'] = f1_score(real, predicted)

    real = np.array([])
    predicted = np.array([])
    for idx, label_name in enumerate(label_names):
        real_ = np.array(labels_by_classes[label_name])
        predicted_ = np.array(predicted_by_classes[label_name])
        real = np.concatenate((real, real_ * (idx + 1)))
        predicted = np.concatenate((predicted, predicted_ * (idx + 1)))

    result_cell['loss'] = losses.avg
    result_cell['accuracy'] = accuracies.avg
    result_cell['f1_micro'] = f1_score(real, predicted, average='micro')
    result_cell['f1_macro'] = f1_score(real, predicted, average='macro')
    meters.add_record(epoch_number, result_cell)


def save_images(dataset, to, n=32):
    for i in range(n):
        img_path = os.path.join(to, 'img_{}.png'.format(i))
        save_image(dataset[i][0], img_path)


def main(train_root, train_csv, val_root, val_csv, epochs: int, batch_size: int,
         num_workers,
         lr, experiment_path, experiment_name, start_from_begin, csv_image_field="images", n_classes=5):
    last_model_path = os.path.join(experiment_path, "last_model.pth")
    train_metrics = utils.Dumper(os.path.join(experiment_path, "train_metrics.json"))
    test_metrics = utils.Dumper(os.path.join(experiment_path, "test_metrics.json"))
    metric_holder = {
        'train': train_metrics,
        'val': test_metrics
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ptm.inceptionv4(num_classes=1000, pretrained='imagenet')
    model.last_linear = nn.Linear(model.last_linear.in_features, n_classes)
    model = ModelWithSigmoidOut(model)

    if start_from_begin:
        epochs_list = [i for i in range(epochs)]
    else:
        epochs_list = [i for i in range(train_metrics.latest_key(), epochs)]
        model.load_state_dict(torch.load(last_model_path))

    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logging.debug('total params')
    logging.debug(total_params)
    logging.debug("model: {}".format(model))

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(299, scale=(0.75, 1.0)),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 * 0.1),
            # transforms.ColorJitter(hue=0.2),
            transforms.ToTensor(),
            # transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(299, scale=(0.75, 1.0)),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 * 0.1),
            # transforms.ColorJitter(hue=0.2),
            transforms.ToTensor(),
            # transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    logging.debug('augmentation: ' + str(data_transforms))
    # image_name, target
    train_ds = CSVDatasetWithName(
        train_root, train_csv, csv_image_field,
        transform=data_transforms['train'], add_extension=".jpg", split=None)
    if val_root is not None:
        val_ds = CSVDatasetWithName(
            val_root, val_csv, csv_image_field,
            transform=data_transforms['val'], add_extension=".jpg", split=None)
    else:
        val_ds = None

    labels = train_ds.target_fields

    datasets = {
        'train': train_ds,
        'val': val_ds,
    }

    data_sampler = None
    shuffle = False
    dl_train = DataLoader(datasets['train'], batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers,
                          sampler=data_sampler)
    if val_root is not None:
        dl_val = DataLoader(datasets['val'], batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers,
                            sampler=None)
    else:
        dl_val = None

    dataloaders = {
        'train': dl_train,
        'val': dl_val,
    }

    criterion = nn.BCELoss()  # because on single image might be multiple classes

    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=0.001)

    #if val_root is not None:
    #    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1,
    #                                                     min_lr=1e-5, patience=10)
    #else:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[25],
                                                   gamma=0.1)
    batches_per_epoch = None

    for epoch in epochs_list:
        logging.debug('train epoch {}/{}'.format(epoch + 1, epochs))
        train_epoch(
            device, model, dataloaders, metric_holder, criterion, optimizer, 'train',
            epoch, epochs,
            labels,
            batches_per_epoch)

        if val_root is not None:
            logging.debug('val epoch {}/{}'.format(epoch + 1, epochs))
            train_epoch(
                device, model, dataloaders, metric_holder, criterion, optimizer, 'val',
                epoch, epochs,
                labels,
                batches_per_epoch)
            logging.debug('-' * 40)

        scheduler.step()

        torch.save(model, last_model_path)


if __name__ == "__main__":
    params = pl.initialize([
        '--train_root', '/Users/nduginets/Desktop',
        '--train_csv', '/Users/nduginets/PycharmProjects/master-diploma/splits/validation.csv',
        "--validate_root", "/Users/nduginets/Desktop",
        "--validate_csv", "/Users/nduginets/PycharmProjects/master-diploma/splits/validation.csv",
        "--epochs", "10",
        "--learning_rate", "0.001",
        "--result_dir", "/Users/nduginets/Desktop",
        "--experiment_name", "tmp",
        "--num_workers", "0",  # stupid Mac os!!!!
        "--batch_size", "7"
    ])

    # params = pl.initialize()

    ex_path = os.path.join(params.result_dir, params.experiment_name)
    main(
        params.train_root,
        params.train_csv,
        params.validate_root,
        params.validate_csv,
        params.epochs,
        params.batch_size,
        params.num_workers,
        params.learning_rate,
        ex_path,
        params.experiment_name,
        params.start_from_begin)
