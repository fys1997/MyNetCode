# This is a sample Python script.
import torch
import torch.nn as nn
import torch.nn.functional as F
import util
import numpy as np
import pandas as pd
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    a=torch.randn(2,2)
    li=nn.Linear(in_features=2,out_features=1)
    b=li(a)
    print(a.shape)
    print(b.shape)









# See PyCharm help at https://www.jetbrains.com/help/pycharm/
