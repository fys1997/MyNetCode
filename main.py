# This is a sample Python script.
import math

import torch

import torch.nn as nn
import torch.nn.functional as F
import util
import numpy as np
import pandas as pd
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def generate_square_subsequent_mask(B, N, T) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    mask_shape = [B, N, 1, T, T]
    with torch.no_grad():
        return torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1)
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    a=pd.read_pickle('data/sensor_graph/adj_mx.pkl')
    b=pd.read_hdf('data/metr-la.h5')
    print("hi")









# See PyCharm help at https://www.jetbrains.com/help/pycharm/
