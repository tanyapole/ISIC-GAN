# Directories
atri_dir = os.path.join(base_path, 'attribute_512p_box/')
image_dir = os.path.join(base_path, 'images_512p/')
segmentation_dir = os.path.join(base_path, 'seg_512p_box/')
output_dir = os.path.join(base_path, 'instance_map_no_border/')
print("atri_dir", atri_dir)
print("image_dir", image_dir)
print("segmentation_dir", segmentation_dir)
print("output_dir", output_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

file_name_arr = []
for file in glob.glob(atri_dir + '*.png'):
    temp = file.split('/')[-1].split('_')
    file_name = temp[0] + '_' + temp[1]
    if file_name not in file_name_arr:
        file_name_arr.append(file_name)


def create_instance_map(family):
    # Create a zero filled base image
    # Load original image
    image = imread(image_dir + family + '.png')
    instance_map = np.zeros(image.shape[:2], dtype=int)
    segments = slic(img_as_float(image), n_segments=1000,
                    slic_zero=True, compactness=1, sigma=2)

    for i, file in enumerate(glob.glob(atri_dir + family + '*.png')):
        # Read Mask
        mask = imread(file)
        type_file = file.split('/')[-1].split('_')[3]
        if i == 0:
            segmentation = imread(segmentation_dir + family + '_segmentation.png', flatten=True)
            last_lesion = 2000
            last_background = 1000
            for v in np.unique(segments):
                union = segmentation[segments == v]
                if float(len(union[union > 0])) / float(len(union)) > 0.5:
                    instance_map[segments == v] = last_lesion
                    last_lesion += 1
                else:
                    instance_map[segments == v] = last_background
                    last_background += 1

        if type_file == 'pigment':  # 3
            last = 3000
        elif type_file == 'negative':  # 4
            last = 4000
        elif type_file.startswith('streaks'):  # 5
            last = 5000
        elif type_file == 'milia':  # 6
            last = 6000
        elif type_file.startswith('globules'):  # 7
            last = 7000
        else:
            print('ERROR: Invalid File Found!!!!')
        # For each superpixel in the selected mask,
        # update the pixel values incrementing the value for each superpixel found.
        for v in np.unique(segments):
            union = mask[segments == v]
            if float(len(union[union > 0])) / float(len(union)) > 0.5:
                instance_map[segments == v] = last
                last += 1
    instance_map = instance_map.astype(np.uint32)
    im = Image.fromarray(instance_map)
    im.save(output_dir + family + '_instance.png')


create_instance_map(file_name_arr[0])
# results = Parallel(n_jobs=8)(delayed(create_instance_map)(family) for family in tqdm(file_name_arr))
# print(results)
