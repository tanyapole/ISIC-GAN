# How to build and train
 
## Prepare dataset before passing into pix2pix
* At the first datasets must be downloaded, in this work we used [ISIC 2018 dataset](https://challenge.isic-archive.com/data/#2018)
* You must download 3 datasets:
  * Training data for tasks 1-2 (10.4 G)
  * Training Ground Truth for task 1 (26 MB) & task 2 (33 MB)
  * Validation Data for tasks 1-2 (228 MB)
  * Validation Ground Truth for task 1 (742 KB) & task 2 (1 MB)
* Create directory named `images`
* Then unpack these zips into `images` directory
* Out of the box already works baseline model, to support model with generated images pix2pix generator must be trained first

### Prepare the environment
* Install conda
* Create conda env `$ conda env create -f envs.yml`

### Prepare dataset to train pix2pix network
* to pass original images into pix2pix model it must be processed into the correseponded format
* go to `dataset-to-pix2pix-data` folder
* modify the 3rd line of `resize-images.sh` by filling in the __absolute__ path to the root of this repository

  E.g. `REPO_DIR="~/master-diploma"` if this repository is located at `~/master-diploma`
* execute bash script with arguments: 
  ```
  $ chmode +x resize-images.sh
  $ DIR=<data-root> ./resize-images.sh -a ISIC2018_Task2_Training_GroundTruth_v3 -s ISIC2018_Task1_Training_GroundTruth -i ISIC2018_Task1-2_Training_Input
  ```

  where `<data-root>` is __absolute__ path of the folder `images`

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

  All script arguments have the same meaning as in the command above
* After training the GAN, synthesize images 
  ```
  $ python3 test.py --name <experiment-name> --dataroot <data_root>/datasets/skin --checkpoints_dir <directory-to-store-temporary-results> --label_nc 8 --how_many 10000 --gpu_id  <gpu-id> --results_dir images/pix2pix_result/
  ```

  All script arguments have the same meaning as in the command above

### Augmentation techniques
* At this step need to create fake images
* go to `bboxes` folder
* execute `python 1_noise_crop.py <path-to-bounding_boxes_metadata.csv> <path-to-folder-images_512p> <base-path-to-storage-fake-images>`  -- this script will create a lot of fake images
* execute `python 2_noise_data_to_pix2pix.py <base-path-to-storage-fake-images>`  -- process images created on the previous step to create images acceptable by pix2pix
* pass generated data throw pix2pix GAN with command: `python3 test.py --name <experiment-name> --dataroot <path-to-lesions-with-masks> --checkpoints_dir <directory-to-storage-temporary-results> --label_nc 8 --how_many 10000 --gpu_id  <gpu-id> --results_dir <result-dir>`
* execute `python 3_create_fake_dataset <base-path-to-storage-fake-images>` -- generates csv files
* at the end you'll get set of folders with different strategies to train and execute


### Prepare data to pass into classification model
* I already split datasets
* use `splits` folder to train model with usual data
* use `splits_boxed` folder to train model with bounding boxes 
* If you want to create custom splitting with scripts from `splits` folder

### Train classification model
Model based on InceptionV4 network
* `cd classificator_network` -- go to the classificator directory
* `pythob train.py --train_root <full-path-to-train-images-folder> --train_csv <full-path-to-train-csv-image> --validate_root <full-path-to-validate-images-folder> --validate_csv <full-path-to-validate-csv-image> --epochs <epochs-count> --result_dir <base-result-directory> --experiment_name <launch-name>` -- execute this code
* after train you'll receive json file with metrics, which contains accuracy, f1 measure, AUC