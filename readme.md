# hello there 

 

# prepare dataset before passing into pix2pix


# Train pix2pix network
* `cd pix2pixHD` -- go to the GAN directory
* `python train.py --name <experiment-name> --dataroot <path-to-lesions-with-masks> --label_nc 8 --checkpoints_dir <directory-to-storage-temporary-results> --gpu_id <gpu-id> --batchSize 4` -- to start train
* `python train.py --name <experiment-name> --dataroot <path-to-lesions-with-masks> --label_nc 8 --checkpoints_dir <directory-to-storage-temporary-results> --gpu_id <gpu-id> --batchSize 4 --continue_train` -- to continue train

# prepare data to pass into classification model
* I already split datasets
* use `splits` folder to train model with usual data
* use `splits_boxed` folder to train model with bounding boxes 


# Train classification model
Model based on InceptionV4 network

* `cd classificator_network` -- go to the classificator directory
* `pythob train.py --train_root <full-path-to-train-images-folder>
  --train_csv <full-path-to-train-csv-image> --validate_root <full-path-to-validate-images-folder> --validate_csv <full-path-to-validate-csv-image> --epochs <epochs-count> --result_dir <base-result-directory> --experiment_name <launch-name>` -- execute this code
