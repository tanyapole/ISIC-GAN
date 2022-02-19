import pandas as pd
import numpy as np
import sys
import os

import glob

if __name__ == "__main__":
    if len(sys.argv) != 3:
        exit(1)
    from_file = sys.argv[1]
    to_file = sys.argv[2]

    from_frame = pd.read_csv(from_file)
    mapping = lambda x: x * 2.0 - 1.0
    from_frame = from_frame.applymap(mapping)
    from_frame.to_csv(to_file, index=False)
