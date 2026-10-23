"""Evaluation and prediction plots for the Vision Transformer."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix


def _plot_predictions(indices, title, color, images, labels, predictions):
    figure, axes = plt.subplots(2, 5, figsize=(12, 5))
    figure.suptitle(title, fontsize=14)
    axes = axes.ravel()

    for axis, index in zip(axes, indices):
        axis.imshow(images[index], cmap="gray")
        axis.set_title(
            f"Actual: {labels[index]} | Guess: {predictions[index]}", color=color
        )
        axis.axis("off")

    for axis in axes[len(indices):]:
        axis.axis("off")
    figure.tight_layout()


def evaluate_model(model, test_loader, device, loss_function, x_test_images, y_test):
    model.eval()
    all_predictions = []
    total_test_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for patches, labels in test_loader:
            patches = patches.to(device)
            labels = labels.to(device)
            logits = model(patches)
            total_test_loss += loss_function(logits, labels).item() * len(patches)
            total_examples += len(patches)
            all_predictions.append(logits.argmax(dim=1).cpu())

    predictions = torch.cat(all_predictions).numpy()
    test_accuracy = np.mean(predictions == y_test)
    print(f"\nTest Loss: {total_test_loss / total_examples:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    class_counts = np.bincount(predictions, minlength=10)
    print("\nSố lượng ảnh dự đoán cho từng class:")
    for digit, count in enumerate(class_counts):
        print(f"Class {digit}: {count}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, digits=4))

    correct_indices = np.where(predictions == y_test)[0][:10]
    incorrect_indices = np.where(predictions != y_test)[0][:10]
    _plot_predictions(
        correct_indices, "10 ảnh dự đoán đúng", "green", x_test_images, y_test, predictions
    )
    _plot_predictions(
        incorrect_indices, "10 ảnh dự đoán sai", "red", x_test_images, y_test, predictions
    )
    plt.show()
    return predictions
