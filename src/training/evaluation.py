import matplotlib.pyplot as plt
import torch


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    correct, total_loss = 0, 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
    n = len(loader.dataset)
    return {"loss": total_loss / n, "acc": correct / n}


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total}


def plot_history(history, path):
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax1.set_xlabel("Epoche")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax2.plot(epochs, [h["val_acc"] for h in history])
    ax2.set_xlabel("Epoche")
    ax2.set_ylabel("Val-Accuracy")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
