# This is a sample Python script.
import torch
import torch.nn as nn
import torch.nn.functional as F
import util
import numpy as np
import pandas as pd
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   df = pd.read_hdf('data/metr-la.h5')
   y=df.index.values
   z=df.index.values.astype("datetime64[D]")
   t=y-z
   x=t/t[1]
   x = np.tile(x, [1, 10, 1]).transpose((2, 1, 0))
   print(x[0,:,:])







# See PyCharm help at https://www.jetbrains.com/help/pycharm/
