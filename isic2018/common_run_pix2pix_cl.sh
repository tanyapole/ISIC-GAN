#!/bin/bash

DEVICES=$1
SPLITS=$(echo $2 | tr ";" "\n")

for split in $SPLITS; do
  ./run_pix2pix_cl.sh "$DEVICES" "$split" "0;1;2;3;4;5;6;7;8;9"
done

