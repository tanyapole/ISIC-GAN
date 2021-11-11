#!/bin/bash
  
isic19_root="/mnt/tank/scratch/nduginets"


val_csv="/nfs/home/nduginets/gan-aug-analysis/splits/isic2019-val.csv"

train_pix2pix_clf="/nfs/home/nduginets/gan-aug-analysis/splits/different_gans/real-pix2pixhd/train.csv"

DEVICES=$1
SPLITS=$(echo $2 | tr ";" "\n")

for split in $SPLITS; do
CUDA_VISIBLE_DEVICES=0 python3 train_comet_csv.py with \
                                train_root=${isic19_root} train_csv=${train_pix2pix_clf} epochs=100\
                                val_root=${isic19_root} val_csv=${val_csv} model_name="inceptionv4" exp_desc="pix2pix"\
                                exp_name="gans.train_pix2pix.inceptionv4.split${split}"
done

#CUDA_VISIBLE_DEVICES=0 python3 train_comet_csv.py with train_root="/mnt/tank/scratch/nduginets"\
#  train_csv="/nfs/home/nduginets/gan-aug-analysis/splits/different_gans/real-pix2pixhd/train.csv"\
#  epochs=100 val_root="/mnt/tank/scratch/nduginets" val_csv="/nfs/home/nduginets/gan-aug-analysis/splits/isic2019-val.csv"\
#  model_name="inceptionv4" exp_desc="pix2pix" exp_name="gans.train_pix2pix.inceptionv4.split0"