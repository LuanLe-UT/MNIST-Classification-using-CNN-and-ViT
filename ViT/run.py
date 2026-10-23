"""Train and evaluate the Vision Transformer on MNIST."""

from copy import deepcopy

import torch
from torch import nn

if __package__:
    from .dataset import load_mnist
    from .evaluate import evaluate_model
    from .model import VisionTransformer
else:
    from dataset import load_mnist
    from evaluate import evaluate_model
    from model import VisionTransformer


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 10
    train_loader, validation_loader, test_loader, x_test_images, y_test = load_mnist(
        batch_size=batch_size,
        patch_size=2,
    )

    model = VisionTransformer(patch_size=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    train_size = len(train_loader.dataset)
    patience = 3
    min_delta = 0.001
    best_validation_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        for patches, labels in train_loader:
            patches = patches.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(patches)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.shape[0]
            total_correct += (logits.argmax(dim=1) == labels).sum().item()

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Loss: {total_loss / train_size:.4f} - "
            f"Accuracy: {total_correct / train_size:.4f}"
        )

        model.eval()
        validation_loss = 0.0
        validation_size = len(validation_loader.dataset)
        with torch.no_grad():
            for patches, labels in validation_loader:
                patches = patches.to(device)
                labels = labels.to(device)
                logits = model(patches)
                validation_loss += loss_function(logits, labels).item() * len(patches)
        validation_loss /= validation_size
        print(f"Validation Loss: {validation_loss:.4f}")

        if validation_loss < best_validation_loss - min_delta:
            best_validation_loss = validation_loss
            best_model_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after epoch {epoch + 1}.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    evaluate_model(
        model,
        test_loader,
        device,
        loss_function,
        x_test_images,
        y_test,
    )


if __name__ == "__main__":
    main()
