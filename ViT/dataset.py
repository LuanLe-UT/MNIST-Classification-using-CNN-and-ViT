"""MNIST loading and image-to-patch preprocessing for the ViT."""

import numpy as np
import torch
from keras.datasets import mnist
from torch.utils.data import DataLoader, TensorDataset, random_split


def split_into_patches(images, patch_size=2):
    """Convert NCHW images into flattened, row-major patches.

    For 28x28 MNIST images and 2x2 patches, this returns 196 tokens per
    image, with four values per token (the flattened 2x2x1 patch).
    """
    if images.ndim != 4:
        raise ValueError("images must have shape (batch, channels, height, width)")

    batch_size, channels, height, width = images.shape
    if height % patch_size or width % patch_size:
        raise ValueError("image height and width must be divisible by patch_size")

    patch_rows = height // patch_size
    patch_columns = width // patch_size
    patches = images.reshape(
        batch_size,
        channels,
        patch_rows,
        patch_size,
        patch_columns,
        patch_size,
    )
    patches = patches.permute(0, 2, 4, 1, 3, 5)
    return patches.reshape(batch_size, patch_rows * patch_columns, -1)


def load_mnist(batch_size=64, validation_fraction=0.1, patch_size=2):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("Train:", x_train.shape)
    print("Test :", x_test.shape)

    x_train_images = x_train.astype(np.float32) / 255.0
    x_test_images = x_test.astype(np.float32) / 255.0
    x_train_tensor = torch.from_numpy(x_train_images).unsqueeze(1)
    x_test_tensor = torch.from_numpy(x_test_images).unsqueeze(1)
    y_train_tensor = torch.from_numpy(y_train.astype(np.int64))
    y_test_tensor = torch.from_numpy(y_test.astype(np.int64))

    train_patches = split_into_patches(x_train_tensor, patch_size)
    test_patches = split_into_patches(x_test_tensor, patch_size)
    full_train_dataset = TensorDataset(train_patches, y_train_tensor)
    validation_size = int(len(full_train_dataset) * validation_fraction)
    train_size = len(full_train_dataset) - validation_size
    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(test_patches, y_test_tensor),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, validation_loader, test_loader, x_test_images, y_test
