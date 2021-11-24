#!/bin/bash
  
train_root="/mnt/tank/scratch/nduginets"
validate_root="/mnt/tank/scratch/nduginets"
result_dir="/mnt/tank/scratch/nduginets/segmentations"
validate_csv="/nfs/home/nduginets/master-diploma/segmentation_splits/validation.csv"

DEVICES=$1
SPLITS=$(echo $2 | tr ";" "\n")

for split in $SPLITS; do

train_csv="/nfs/home/nduginets/master-diploma/segmentation_splits/baseline/train_${split}.csv"

CUDA_VISIBLE_DEVICES=$DEVICES python3 train.py \
                                --train_root ${train_root} --train_csv=${train_csv} --epochs=100\
                                --validate_root=${validate_root} --validate_csv=${validate_csv} --learning_rate 0.001\
                                --result_dir ${result_dir} --experiment_name "segmentation_${split}"\
                                --batch_size 5
done

# CUDA_VISIBLE_DEVICES=0 python3 train_comet_train_comet_csv.pycsv.py with train_root="/mnt/tank/scratch/nduginets" train_csv="/nfs/home/nduginets/gan-aug-analysis/splits/percentages/1_0/train_0.csv" epochs=100 val_root="/mnt/tank/scratch/nduginets" val_csv="/nfs/home/nduginets/gan-aug-analysis/splits/isic2019-val.csv" model_name="inceptionv4" exp_desc="Real" exp_name="gans.train_Real.inceptionv4.split0"