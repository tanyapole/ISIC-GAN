import argparse
from torch.utils.data import Dataset, DataLoader, sampler
import pandas as pd
import os
import numpy as np
from pathlib import Path
import torch
from tqdm.auto import tqdm
import torchvision.transforms.functional as TFF
import torchvision.transforms as TF
from PIL import Image
import torchvision.models
import torch.nn as nn
import pretrainedmodels as ptm
import wandb
import sklearn.metrics

ATTRIBUTES = ['globules', 'milia_like_cyst', 'negative_network', 'pigment_network', 'streaks']

def create_model(name):
    num_classes = len(ATTRIBUTES)
    if name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'resnext50':
        model = torchvision.models.resnext50_32x4d(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'inceptionv4':
        model = ptm.inceptionv4(num_classes=1000, pretrained='imagenet')
        model.last_linear = nn.Linear(model.last_linear.in_features, num_classes)
    return model

def create_optimizer(model, lr):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)

def read_image(pt, size):
    img = Image.open(pt)
    H,W,_ = np.array(img).shape
    img2 = TFF.center_crop(img, min(H,W))
    img3 = TFF.resize(img2, size)
    return img3
    
class CsvDataset(Dataset):
    def __init__(self, csv_path, repo_fldr, size, trn_tfm=None, preload=False):
        assert os.path.exists(csv_path), 'csv doesnot exist'
        csv = pd.read_csv(csv_path)
        self.labels = csv[ATTRIBUTES].values
        self.labels = torch.tensor(self.labels).float()
        self.preload = preload
        images = [repo_fldr / pt for pt in csv.images.values]
        if self.preload:
            self.images = [read_image(pt, size) for pt in tqdm(images, desc='Load data')]
        else:
            self.images = images
            self.size = size
        self.trn_tfm = trn_tfm
        self.to_tensor = TF.Compose([TF.ToTensor(), TF.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                
            
    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        if self.preload:
            image = self.images[i]
        else:
            image = read_image(self.images[i], self.size)
        if self.trn_tfm is not None:
            image = self.trn_tfm(image)
        return self.to_tensor(image), self.labels[i]
#         return image, self.labels[i]

class Collector:
    def __init__(self):
        self.vals = []
    def put(self, vals):
        self.vals.append(vals.detach().cpu().numpy())
    def get(self):
        if len(self.vals[0].shape):
            return np.concatenate(self.vals, axis=0)
        else:
            return np.stack(self.vals, axis=0)

def get_train_tfm():
    return TF.Compose([
                TF.RandomHorizontalFlip(),
                TF.RandomVerticalFlip(),
                # TF.RandomResizedCrop(299, scale=(0.75, 1.0)),
                TF.RandomRotation(45),
                TF.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 * 0.1),
                # transforms.ColorJitter(hue=0.2),
            ])

def main(args):
    args.repo_fldr = Path(args.repo_fldr)
    trn_ds = CsvDataset(args.trn_csv_path, args.repo_fldr, size=args.image_size, preload=True, trn_tfm=get_train_tfm())
    val_ds = CsvDataset(args.val_csv_path, args.repo_fldr, size=args.image_size, preload=True)
    trn_dl = DataLoader(trn_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = create_model(args.model).cuda()
    optimizer = create_optimizer(model, args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    run = wandb.init(project='ISIC-GAN-cl-HP-search', config=args.__dict__)

    for epoch in tqdm(list(range(args.num_epochs)), desc='Epoch'):
        # train
        model.train()
        trn_losses = Collector()
        for img, lbl in tqdm(trn_dl, desc='Train', leave=False):
            optimizer.zero_grad()
            out = model(img.cuda())
            loss = loss_fn(out, lbl.cuda())
            loss.backward()
            optimizer.step()
            trn_losses.put(loss)
        
        # valid
        model.eval()
        lbls, preds, val_losses = Collector(), Collector(), Collector()
        with torch.no_grad():
            for img, lbl in tqdm(val_dl, desc='Valid', leave=False):
                out = model(img.cuda())
                loss = loss_fn(out, lbl.cuda())
                lbls.put(lbl.int())
                preds.put((out > 0).int())
                val_losses.put(loss)

        log = {'epoch': epoch, 'loss/trn': trn_losses.get().mean(), 'loss/val': val_losses.get().mean()}
        
        lbls = lbls.get()
        preds = preds.get()
        
        f1s = []
        for m, attr in enumerate(ATTRIBUTES):
            f1 = sklearn.metrics.f1_score(lbls[:,m], preds[:,m])
            f1s.append(f1)
            log[f'f1/{attr}'] = f1
        log['f1/macro'] = np.array(f1s).mean()
        
        run.log(log)
    
    run.finish()
    
def _get_debug_args():
    args = type('', (), {})()
    args.batch_size = 2 # 64
    args.cuda_idx = 1
    args.model='resnet50'
    args.lr = 1e-3
    args.num_epochs = 1
    args.image_size = 384
    args.repo_fldr = '/mnt/tank/scratch/tpolevaya/my_work/GANs/master-diploma'
    args.trn_csv_path = '../splits/small.csv' # '../splits/baseline_bussio/train_0.csv'
    args.val_csv_path = '../splits/small.csv' # '../splits/validation_skin_lesion.csv'
    return args

def _get_cmd_args():
    parser = argparse.ArgumentParser('Classification HP search')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--cuda_idx', type=int, required=True)
    parser.add_argument('--model', type=str, choices=['resnet50', 'resnext50', 'inceptionv4'], required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--image_size', type=int, default=384)
    parser.add_argument('--repo_fldr', type=str, required=True)
    parser.add_argument('--trn_csv_path', type=str, required=True)
    parser.add_argument('--val_csv_path', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = _get_cmd_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_idx)
    os.environ["WANDB_SILENT"] = "True"
    main(args)