#!/bin/bash


if [ -z "${DIR}" ]
then
  echo "required DIR argument"
  exit 1
fi
BASE_DIR="${DIR}"
echo "$BASE_DIR"
echo ""

python instance_map.py "$BASE_DIR"
mkdir "$BASE_DIR/instance_map"
cd "$BASE_DIR/instance_map"
find "$BASE_DIR/instance_map_no_border" -name '*.png' -exec sh -c 'echo "{}"; convert "{}" -resize 1024x512\> -size 1024x512 xc:black +swap -gravity center -composite `basename "{}" .png`.png' \;

