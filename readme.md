# How to build and train

## Prepare the environment
* Install conda
* Create conda env `$ conda env create -f envs.yml`

## Prepare Data
 
### Download original dataset
* At the first datasets must be downloaded, in this work we used [ISIC 2018 dataset](https://challenge.isic-archive.com/data/#2018)
* You must download 3 datasets:
  * Training data for tasks 1-2 (10.4 G)
  * Training Ground Truth for task 1 (26 MB) & task 2 (33 MB)
  * Validation Data for tasks 1-2 (228 MB)
  * Validation Ground Truth for task 1 (742 KB) & task 2 (1 MB)
* Create directory named `images`
* Then unpack these zips into `images` directory
* Out of the box already works baseline model, to support model with generated images pix2pix generator must be trained first

### Prepare dataset for GAN
* go to `dataset-to-pix2pix-data` folder
* modify the 3rd line of `resize-images.sh` by filling in the __absolute__ path to the root of this repository

  E.g. `REPO_DIR="~/master-diploma"` if this repository is located at `~/master-diploma`
* execute bash script with arguments:
  ```
  $ chmod +x resize-images.sh
  $ DIR=<data-root> ./resize-images.sh -a ISIC2018_Task2_Training_GroundTruth_v3 -s ISIC2018_Task1_Training_GroundTruth -i ISIC2018_Task1-2_Training_Input
  ```

  where `<data-root>` is __absolute__ path of the folder `images`

### Prepare bounding boxes for GAN
* go to `bounding_boxes` folder
* modify the 3rd line of `resize-images.sh` by filling in the __absolute__ path to the root of this repository

  E.g. `REPO_DIR="~/master-diploma"` if this repository is located at `~/master-diploma`
* execute bash script with arguments: 
  ```
  $ chmod +x process_images.sh
  $ DIR=<data-root> ./process_images.sh -a ISIC2018_Task2_Training_GroundTruth_v3 -s ISIC2018_Task1_Training_GroundTruth -i ISIC2018_Task1-2_Training_Input
  ```

  where `<data-root>` is __absolute__ path of the folder `images`

## Train GAN

### Train pix2pix network
* go to the GAN directory `$ cd pix2pixHD`
* Start training a GAN
  ```
  $ python train.py --name <experiment-name> --dataroot <data_root>/datasets/skin --label_nc 8 --checkpoints_dir <directory-to-store-temporary-results> --gpu_id <gpu-id> --batchSize 4
  ```

  where 
  * `<experiment-name>` is the name by which the trained model will be identified by other scripts
  * `<data-root>` is __absolute__ path of the folder `images`
  * `<directory-to-store-temporary-results>` is name of the directory that will be created by the script under `pix2pixHD` and where training metadata will be stored
  * <gpu-id> is int number - the model will be trained on `cuda:<gpu-id>`
* If needed, resume training a GAN
  ```
  $ python train.py --name <experiment-name> --dataroot <data_root>/datasets/skin --label_nc 8 --checkpoints_dir <directory-to-store-temporary-results> --gpu_id <gpu-id> --batchSize 4 --continue_train
  ```

### Synthesize new images
  All script arguments have the same meaning as in the command above
* After training the GAN, synthesize images 
  ```
  $ python test.py --name <experiment-name> --dataroot <data_root>/datasets/skin --checkpoints_dir <directory-to-store-temporary-results> --label_nc 8 --how_many 10000 --gpu_id  <gpu-id> --results_dir <data_root>/pix2pix_result/ --phase train
  ```

  where 
  * `<experiment-name>` is the same as in training the GAN
  * `<data-root>` is __absolute__ path of the folder `images`
  * `<directory-to-store-temporary-results>` is the same as in training the GAN
  * <gpu-id> is int number - image synthesis will be performed on `cuda:<gpu-id>`

### Augmentation techniques
* At this step need to create fake images
* go to `bboxes` folder
* execute `python 1_noise_crop.py <path-to-bounding_boxes_metadata.csv> <path-to-folder-images_512p> <base-path-to-storage-fake-images>`  -- this script will create a lot of fake images
* execute `python 2_noise_data_to_pix2pix.py <base-path-to-storage-fake-images>`  -- process images created on the previous step to create images acceptable by pix2pix
* pass generated data throw pix2pix GAN with command: `python3 test.py --name <experiment-name> --dataroot <path-to-lesions-with-masks> --checkpoints_dir <directory-to-storage-temporary-results> --label_nc 8 --how_many 10000 --gpu_id  <gpu-id> --results_dir <result-dir>`
* execute `python 3_create_fake_dataset <base-path-to-storage-fake-images>` -- generates csv files
* at the end you'll get set of folders with different strategies to train and execute


## Train classification model
### Prepare data to pass into classification model
* I already split datasets
* use `splits` folder to train model with usual data
* use `splits_boxed` folder to train model with bounding boxes 
* If you want to create custom splitting with scripts from `splits` folder

### Train classification model
Model based on InceptionV4 network
* go to the classificator directory `$ cd classificator_network`
* run trainig classifier
  ```
  $ python train.py --train_root <data-root-parent> --train_csv <full-path-to-train-csv-image> --validate_root <data-root-parent> --validate_csv <full-path-to-validate-csv-image>  --result_dir <base-result-directory> --experiment_name <launch-name> --epochs 100 --num_workers 0 --batch_size 32 --learning_rate 0.001 --gpu_id <gpu_id>
  ```

  where
  * `<data-root-parent>` is __absolute__ path of the folder parent folder of `images` folder
  * `<full-path-to-train-csv-image>` is __absolute__ path of the csv with train data
  * `<full-path-to-validate-csv-image>` is __absolute__ path of the csv with test data
  * `<base-result-directory>` is relative or absolute path of the folder where results will be stored inside
  * `<launch-name>` is experiment name
  * `<gpu_id>` is the id of gpu for training the model

  Note
  1. results will be saved as a json file with metrics, which contains accuracy, f1 measure, AUC values under `<base-result-directory>/<launch-name>`
  2. final model will be saved under `<base-result-directory>/<launch-name>/last_model.pth`
  
  Different running options
  * to run with Bissoto et al.'s train-test split use
    * `<full-path-to-train-csv-image>`=`<repo-root>/splits/baseline_bussio/train_<i>.csv`, where `<i>` is the number of run = 0..9
    * `<full-path-to-validate-csv-image>`=`<repo-root>/splits/validation_skin_lesion.csv`
  * to run with original train-test split use
    * `<full-path-to-train-csv-image>`=`<repo-root>/splits/baseline/train_<i>.csv`, where `<i>` is the number of run = 0..9
    * `<full-path-to-validate-csv-image>`=`<repo-root>/splits/validation.csv`
  
  where `<repo-root>` is __absolute__ path of the root of this repository

## Utility files
* generated data splits creation
  ```
  $ cd bounding_boxes
  $ python create_generated_split.py --data-root <data-root> --generated-data-folder <generated-data-folder> --ratios 0.2 0.5 0.8 1.0 --seeds 0 1 2 3 4 5 7 8 9
  ```

  where
  <data-root> is __absolute__ path of the `images` folder
  <generated-data-folder> is __absolute__ path to the folder that contains generated images
