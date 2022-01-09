#!/bin/bash

if [ -z "${DIR}" ]
then
  echo "required DIR argument"
  exit 1
fi
BASE_DIR="${DIR}"
echo "$BASE_DIR"
echo ""

while getopts a:s:i: option
do
	case "${option}"
		in
		a) ATTRI_DIR=${BASE_DIR}/${OPTARG};;
		s) SEG_DIR=${BASE_DIR}/${OPTARG};;
		i) IMAGE_DIR=${BASE_DIR}/${OPTARG};;
	esac
done

echo "Attribute directory: $ATTRI_DIR"
echo "Segmentation directory: $SEG_DIR"
echo "Image directory: $IMAGE_DIR"
echo ""

attribute_resized="${BASE_DIR}/attribute_resized"
segmentation_resized="${BASE_DIR}/segmentation_resized"
image_resized="${BASE_DIR}/image_resized"

echo "attribute_resized: $attribute_resized"
echo "segmentation_resized: $segmentation_resized"
echo "image_resized: $image_resized"
echo ""

mkdir "$BASE_DIR/images_512p"
cd "$BASE_DIR/images_512p"
find "$IMAGE_DIR" -name '*jpg' -exec sh -c 'echo "{}"; convert "{}" -resize 1024x512\> `basename "{}" .jpg`.png' \;


mkdir "$BASE_DIR/attribute_512p"
cd "$BASE_DIR/attribute_512p"
find "$ATTRI_DIR" -name '*.png' -exec sh -c 'echo "{}"; convert "{}" -resize 1024x512 `basename "{}" .png`.png' \;
python ~/master-diploma/bounding_boxes/create_bounding_box.py "$BASE_DIR" attribute_512p attribute_512p_box


mkdir "$BASE_DIR/seg_512p"
cd "$BASE_DIR/seg_512p"
find "$SEG_DIR" -name '*.png' -exec sh -c 'echo "{}"; convert "{}" -resize 1024x512 `basename "{}" .png`.png' \;
python ~/master-diploma/bounding_boxes/create_bounding_box.py "$BASE_DIR" seg_512p seg_512p_box

python ~/master-diploma/bounding_boxes/instance_map.py "$BASE_DIR"
mkdir "$BASE_DIR/instance_map"
cd "$BASE_DIR/instance_map"
find "$BASE_DIR/instance_map_no_border" -name '*.png' -exec sh -c 'echo "{}"; convert "{}" -resize 1024x512\> -size 1024x512 xc:black +swap -gravity center -composite `basename "{}" .png`.png' \;
