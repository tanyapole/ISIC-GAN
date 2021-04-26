import sys
import os

if __name__ == "__main__":
    print(sys.argv)
    a = "a"
    b = "b"
    res = os.path.join(a, b)
    print(res)