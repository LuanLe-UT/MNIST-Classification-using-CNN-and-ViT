import numpy as np
import torch
from keras.datasets import mnist
from torch.utils.data import DataLoader, TensorDataset, random_split


def load_mnist(batch_size=64, validation_fraction=0.1):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print("Train:", x_train.shape)
    print("Test:", x_test.shape)

    x_test_images = x_test.astype(np.float32) / 255.0
    x_train_tensor = torch.from_numpy(
        x_train.astype(np.float32) / 255.0
    ).unsqueeze(1)
    x_test_tensor = torch.from_numpy(x_test_images).unsqueeze(1)
    y_train_tensor = torch.from_numpy(y_train.astype(np.int64))
    y_test_tensor = torch.from_numpy(y_test.astype(np.int64))

    full_train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    validation_size = int(len(full_train_dataset) * validation_fraction)
    train_size = len(full_train_dataset) - validation_size
    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(x_test_tensor, y_test_tensor),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, validation_loader, test_loader, x_test_images, y_test