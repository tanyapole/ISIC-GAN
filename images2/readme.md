# Segm maps
1. go to this folder `$ cd images2`
1. create resized images by calling [script](1_resize_images.sh)
1. create instance maps `$ python instance_map.py`
1. combine masks into conditions `$ python combine_masks.py`
1. pad all inputs to 1024x512 by calling [script](pad.sh)
1. prepare dataset for Pix2PixHD `$ python split.py`

# Cut synthesiszed images
To cut synthesised images `$python cut.py --source <fldr-with-synthesized_imgs> --target <target-fldr>`

# Save augmented masks
`$ python save_augmented_masks.py`

# Bboxes
1. to view bboxes on top of derm attributes contours run from repo root `$python show_bboxes.py`
1. to create bboxes for resized images run from repo root `$python build_bboxes.py`
1. go to this folder `$ cd images2`
1. combine masks into conditions `$ python combine_masks.py --bbox`
1. pad all inputs to 1024x512 by calling [script](pad_bboxes.sh)
