if [ -z "${DIR}" ]
then
  echo "required DIR argument"
  exit 1
fi
BASE_DIR="${DIR}"
echo "$BASE_DIR"
echo ""


mkdir "$BASE_DIR/images_512p"
cd "$BASE_DIR/images_512p"
find "$IMAGE_DIR" -name '*jpg' -exec sh -c 'echo "{}"; convert "{}" -resize 1024x512\> `basename "{}" .jpg`.png' \;


mkdir "$BASE_DIR/attribute_512p"
cd "$BASE_DIR/attribute_512p"
find "$ATTRI_DIR" -name '*.png' -exec sh -c 'echo "{}"; convert "{}" -resize 1024x512 `basename "{}" .png`.png' \;


mkdir "$BASE_DIR/seg_512p"
cd "$BASE_DIR/seg_512p"
find "$SEG_DIR" -name '*.png' -exec sh -c 'echo "{}"; convert "{}" -resize 1024x512 `basename "{}" .png`.png' \;
