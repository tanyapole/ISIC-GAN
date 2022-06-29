# How to build and train
 
## Prepare dataset before passing into pix2pix
* At the first datasets must be downloaded, in this work we used [ISIC 2018 dataset](https://challenge.isic-archive.com/data/#2018)
* You must download 3 datasets:
  * Training data (10.4 G)
  * Training Ground Truth (33 MB)
  * Validation Data (228 MB)
  * Validation Ground Truth (1 MB)
* Create directory named `images`
* Then unpack these zips into `images` directory
* Out of the box already works baseline model, to support model with generated images pix2pix generator must be trained first

### Prepare dataset to train pix2pix network
* to pass original images into pix2pix model it must be processed into the correseponded format
* go to `dataset-to-pix2pix-data` folder
* execute bash script with arguments: `DIR=<full-path-to-folders> resize-images.sh -a <relative-parth-to-attribute-dir> -s <relative-parth-to-segmentation-dir> -i <relative-parth-to-images-dir>`

### Train pix2pix network
* `cd pix2pixHD` -- go to the GAN directory
* `python train.py --name <experiment-name> --dataroot <path-to-lesions-with-masks> --label_nc 8 --checkpoints_dir <directory-to-storage-temporary-results> --gpu_id <gpu-id> --batchSize 4` -- this command starts train
* `python train.py --name <experiment-name> --dataroot <path-to-lesions-with-masks> --label_nc 8 --checkpoints_dir <directory-to-storage-temporary-results> --gpu_id <gpu-id> --batchSize 4 --continue_train` -- this command continues train
* When the pix2pix model was trained, need to generate synthesized images 
* `python3 test.py --name <experiment-name> --dataroot <path-to-lesions-with-masks> --checkpoints_dir <directory-to-storage-temporary-results> --label_nc 8 --how_many 10000 --gpu_id  <gpu-id> --results_dir images/pix2pix_result/` -- this script will create generated images

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