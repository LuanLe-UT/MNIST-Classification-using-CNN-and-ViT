import torch
from torch import nn

if __package__:
    from .layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, relu
else:
    from layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, relu


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(1, 8, 3)
        self.pool1 = MaxPool2D(2)
        self.conv2 = Conv2D(8, 16, 3)
        self.pool2 = MaxPool2D(2)
        self.flatten = Flatten()
        self.dense1 = Dense(5 * 5 * 16, 128)
        self.dropout = Dropout(0.3)
        self.dense2 = Dense(128, 10)

    def forward(self, x):
        x = self.pool1(relu(self.conv1(x)))
        x = self.pool2(relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(relu(self.dense1(x)))
        return self.dense2(x)
