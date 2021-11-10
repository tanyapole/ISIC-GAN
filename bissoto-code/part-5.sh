#!/bin/bash


if [ -z "${DIR}" ]
then
  echo "required DIR argument"
  exit 1
fi
BASE_DIR="${DIR}"
echo "$BASE_DIR"
echo ""

python ~/master-diploma/select_train_test.py "$BASE_DIR"