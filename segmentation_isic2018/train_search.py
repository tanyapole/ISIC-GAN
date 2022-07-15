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
import segmentation_models_pytorch as smp
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


ATTRIBUTES = ['globules', 'milia_like_cyst', 'negative_network', 'pigment_network', 'streaks']

def create_model(name):
    num_classes = len(ATTRIBUTES)
    return smp.Unet(name, encoder_weights='imagenet', classes=len(ATTRIBUTES))

def create_optimizer(model, lr):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)

def read_image(pt, size, is_image=True):
    img = Image.open(pt)
    H,W, = np.array(img).shape[:2]
    img2 = TFF.center_crop(img, min(H,W))
    interp_mode = Image.BILINEAR if is_image else Image.NEAREST
    img3 = TFF.resize(img2, size, interpolation=interp_mode)
    return img3
    
class CsvDataset(Dataset):
    def __init__(self, csv_path, repo_fldr, size, trn_tfm=None, preload=False):
        assert os.path.exists(csv_path), 'csv doesnot exist'
        csv = pd.read_csv(csv_path)
        self.preload = preload
        repo_fldr = Path(repo_fldr)
        images = [repo_fldr / pt for pt in csv.images.values]
        masks = {attr: [repo_fldr / pt for pt in csv[attr].values] for attr in ATTRIBUTES}
        if self.preload:
            self.images = [read_image(pt, size, is_image=True) for pt in tqdm(images, desc='Load data')]
            self.masks = {
                attr: [
                    read_image(pt, size, is_image=False) for pt in tqdm(masks[attr], desc=f'Load data {attr}', leave=False)
                ] for attr in ATTRIBUTES
            }
        else:
            raise NotImplementedError()
        self.trn_tfm = trn_tfm
        self.to_tensor = TF.Compose([TF.ToTensor(), TF.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                
            
    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        if self.preload:
            image = np.array(self.images[i])
            masks = np.stack([np.array(self.masks[attr][i]) for attr in ATTRIBUTES], axis=-1)
        else:
            raise NotImplementedError()
        if self.trn_tfm is not None:
            segmap = SegmentationMapsOnImage(masks, shape=image.shape)
            res = self.trn_tfm(image=image, segmentation_maps=segmap)
            image, masks = res[0], res[1].arr
        masks = (torch.tensor(np.moveaxis(masks, -1, 0)) > 0).float()
        return self.to_tensor(image), masks
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
    return iaa.Sequential([
        iaa.Fliplr(p=0.5),
        iaa.Flipud(p=0.5),
        # TF.RandomResizedCrop(299, scale=(0.75, 1.0)),
        iaa.Rotate(rotate=(-45,45)),
        iaa.AddToBrightness(),
        iaa.AddToHueAndSaturation((-50, 50))
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

    run = wandb.init(project='ISIC-GAN-segm-HP-search', config=args.__dict__)

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
        intersections, unions, val_losses = Collector(), Collector(), Collector()
        with torch.no_grad():
            for img, lbl in tqdm(val_dl, desc='Valid', leave=False):
                out = model(img.cuda())
                loss = loss_fn(out, lbl.cuda())
                val_losses.put(loss)

                out = out.detach().cpu()
                pred = (out > 0).float()                
                intersection = (lbl * pred).sum(dim=(2,3))
                union = ((lbl + pred) > 0).float().sum(dim=(2,3))
                intersections.put(intersection)
                unions.put(union)

        log = {'epoch': epoch, 'loss/trn': trn_losses.get().mean(), 'loss/val': val_losses.get().mean()}

        IoUs = intersections.get().sum(axis=0) / unions.get().sum(axis=0)
        for m, attr in enumerate(ATTRIBUTES):
            log[f'IoU/{attr}'] = IoUs[m]
        log['IoU/macro'] = IoUs.mean()
        
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
    parser.add_argument('--model', type=str, choices=['resnet50', 'resnext50_32x4d', 'resnet34'], required=True)
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