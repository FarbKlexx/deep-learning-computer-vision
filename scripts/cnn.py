import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import v2

from src.utils.device import get_device
from scripts.load_dataset import make_loaders

if __name__ == "__main__":
    # Init
    device = get_device()          # "cuda" | "mps" | "cpu"
    print("Selected device: ", device)

    # Dataset: tanganke/eurosat (https://huggingface.co/datasets/tanganke/eurosat)
    # label: The label for the image, which is one of 10 classes:
    #   0: annual crop land
    #   1: forest

    #   2: brushland or shrubland
    #   3: highway or road
    #   4: industrial buildings or commercial buildings
    #   5: pasture land
    #   6: permanent crop land
    #   7: residential buildings or homes or apartments
    #   8: river
    #   9: lake or sea

    num_classes = 10 # defined by the amount of labels

    ## CNN
    weights = models.ResNet50_Weights.IMAGENET1K_V2 # pre-trained weights
    model = models.resnet50(weights=weights) # cnn that uses the weights
    print("Weights loaded.")

    # Preprocessing Pipeline
    eval_tf = weights.transforms()
    train_tf = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(224),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("Finished preprocessing.")

    # Data and Head
    train_dl, val_dl, test_dl, class_names = make_loaders(train_tf, eval_tf)
    model.fc = nn.Linear(model.fc.in_features, num_classes) # reduce output features to fit to the 10 labels in the dataset
    
    # komplettes Netz, Backbone mit kleinerer LR
    model.requires_grad_(True)
    head = list(model.fc.parameters())
    backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": 1e-4},
        {"params": head, "lr": 1e-3},
    ])  

    model.to(device) # move model to the selected device
    criterion = nn.CrossEntropyLoss()
    epochs = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def train_one_epoch(model, loader):
        model.train()
        total = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                loss = criterion(model(images), labels)
            loss.backward()                  # außerhalb von autocast
            optimizer.step()
            total += loss.item() * images.size(0)
        return total / len(loader.dataset)

    @torch.no_grad()
    def evaluate(model, loader):
        model.eval()
        correct = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(images)
            correct += (logits.argmax(dim=1) == labels).sum().item()
        return correct / len(loader.dataset)

    print("Starting training:" )
    for epoch in range(epochs):
        print("Epoch ", epoch, "out of ", epochs)
        loss = train_one_epoch(model, train_dl)
        acc = evaluate(model, val_dl)
        scheduler.step()
        print(f"epoch {epoch}  loss {loss:.3f}  val_acc {acc:.3f}", flush=True)

    print(f"test_acc {evaluate(model, test_dl):.3f}")
    if device == "cuda":
        print(f"peak_vram {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    elif device == "mps":  # MPS kennt keinen Peak-Wert, nur den aktuellen Verbrauch
        print(f"mps_memory {torch.mps.driver_allocated_memory() / 1e9:.1f} GB")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"{trainable:,} / {total:,}")

    torch.save(model.state_dict(), "results/station1/resnet50_ft.pt")
    model.load_state_dict(torch.load("results/station1/resnet50_ft.pt", map_location=device))