import pandas as pd
import numpy as np
import sys
import os

import glob

if __name__ == "__main__":
    sys.argv = ["", "/Users/nduginets/PycharmProjects/master-diploma/GAN_to_box/test_data/isic_2018_boxes.csv",
                "/Users/nduginets/PycharmProjects/master-diploma/GAN_to_box/test_data/isic_2018_boxes_shifted.csv"]

    if len(sys.argv) != 3:
        exit(1)
    from_file = sys.argv[1]
    to_file = sys.argv[2]

    from_frame = pd.read_csv(from_file)
    headers = from_frame.columns[1:]
    indexes = list(from_frame.index)
    for h in headers:
        for i in indexes:
            if "size" in h:
                from_frame.at[i, h] *= 1 # here I won't mult by 2 because tanh must be in range (-1, 1)
            else:
                from_frame.at[i, h] = from_frame.at[i, h] * 2 - 1
    from_frame.to_csv(to_file, index=False)
