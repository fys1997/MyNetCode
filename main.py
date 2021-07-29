# This is a sample Python script.
import torch
import torch.nn as nn
import torch.nn.functional as F
import util
import numpy as np
import pickle
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   X=torch.randn(1,2,3,2)
   Y=X.permute(1,0,2,3).contiguous()
   print(Y.shape)
   print(X.shape)







# See PyCharm help at https://www.jetbrains.com/help/pycharm/
