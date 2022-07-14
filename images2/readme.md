1. create resized images by calling [script](1_resize_images.sh)
1. create instance maps `$ python instance_map.py`
1. combine masks into conditions `$ python combine_masks.py`
1. pad all inputs to 1024x512 by calling [script](pad.sh)
1. prepare dataset for Pix2PixHD `$ python split.py`


To cut synthesised images `$python cut.py --source <fldr-with-synthesized_imgs> --target <target-fldr>`