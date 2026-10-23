import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def relu(x):
    return F.relu(x)


def relu_derivative(x):
    return (x > 0).to(dtype=x.dtype)


def softmax(x):
    shifted = x - torch.max(x, dim=1, keepdim=True).values
    exp_values = torch.exp(shifted)
    return exp_values / torch.sum(exp_values, dim=1, keepdim=True)


class Conv2D(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.randn(output_channels, input_channels, kernel_size, kernel_size)
            * np.sqrt(2.0 / (input_channels * kernel_size * kernel_size))
        )
        self.bias = nn.Parameter(torch.zeros(output_channels))

    def forward(self, x):
        kernel = self.kernel_size
        patches = F.unfold(x, kernel_size=kernel)
        output = self.weight.flatten(1) @ patches
        output = output + self.bias[None, :, None]
        output_height = x.shape[2] - kernel + 1
        output_width = x.shape[3] - kernel + 1
        return output.reshape(
            x.shape[0],
            self.weight.shape[0],
            output_height,
            output_width,
        )


class MaxPool2D(nn.Module):
    def __init__(self, pool_size=2):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, x):
        batch, channels, height, width = x.shape
        size = self.pool_size
        output_height = height // size
        output_width = width // size
        cropped = x[:, :, :output_height * size, :output_width * size]
        windows = cropped.reshape(batch,channels,output_height,size,output_width,size,).permute(0, 1, 2, 4, 3, 5)
        return windows.amax(dim=(-1, -2))


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class Dropout(nn.Module):
    def __init__(self, probability=0.3):
        super().__init__()
        if not 0 <= probability < 1:
            raise ValueError("probability must be in the range [0, 1)")
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        mask = (torch.rand_like(x) >= self.probability).to(x.dtype)
        return x * mask / (1 - self.probability)


class Dense(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        )
        self.bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x):
        return x @ self.weight + self.bias
