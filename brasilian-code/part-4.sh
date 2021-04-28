#!/bin/bash


if [ -z "${DIR}" ]
then
  echo "required DIR argument"
  exit 1
fi
BASE_DIR="${DIR}"
echo "$BASE_DIR"
echo ""


mkdir -p "$BASE_DIR/datasets/skin"
mv "$BASE_DIR/instance_map" "$BASE_DIR/datasets/skin/"
mv "$BASE_DIR/semantic_map" "$BASE_DIR/datasets/skin/"

mkdir -p "$BASE_DIR/datasets/skin/test_label"
mkdir -p "$BASE_DIR/datasets/skin/test_inst"
mkdir -p "$BASE_DIR/datasets/skin/test_img"
mv "$BASE_DIR/datasets/skin/instance_map" "$BASE_DIR/datasets/skin/train_inst"
mv "$BASE_DIR/datasets/skin/semantic_map" "$BASE_DIR/datasets/skin/train_label"
mv "$BASE_DIR/image_resized" "$BASE_DIR/datasets/skin/"
mv "$BASE_DIR/datasets/skin/image_resized" "$BASE_DIR/datasets/skin/train_img"