import torch
import torch.nn as nn
class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.__fc1 = nn.Linear(3, 100)
        self.__fc2 = nn.Linear(100, 3)
        self.__activate = nn.LeakyReLU()

    def forward(self, x):
        x = self.__fc1(x)
        x = self.__activate(x)
        x = self.__fc2(x)
        return x
class Regressor_sig(nn.Module):
    def __init__(self):
        super().__init__()
        self.__fc1 = nn.Linear(3, 100)
        self.__fc2 = nn.Linear(100, 3)
        self.__activate = nn.LeakyReLU()
        self.__sig = nn.Sigmoid()

    def forward(self, x):
        x = self.__fc1(x)
        x = self.__activate(x)
        x = self.__fc2(x)
        x = self.__sig(x)
        return x