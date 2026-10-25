import torch
from torch import nn
from torchvision import models
from torchvision.transforms import v2

from src.utils.device import get_device
from scripts.load_dataset import make_loaders

if __name__ == "__main__":
    # Init
    device = get_device() # "cuda" | "mps" | "cpu"
    print("Selected device: ", device)

    ## CNN
    weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2 # pre-trained weights
    model = models.resnext50_32x4d(weights=weights) # cnn that uses the weights
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
    train_dl, val_dl, test_dl, class_names = make_loaders(train_tf, eval_tf) # load the data
    num_classes = len(class_names) # defined by the amount of labels in the dataset
    model.fc = nn.Linear(model.fc.in_features, num_classes) # reduce output features to fit to the amount of labels in the dataset
    
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

    # Training
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

    # Evaluation
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

    # Evaluate on Test-Set
    print(f"test_acc {evaluate(model, test_dl):.3f}")

    if device == "cuda":
        print(f"peak_vram {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    elif device == "mps":  # MPS kennt keinen Peak-Wert, nur den aktuellen Verbrauch
        print(f"mps_memory {torch.mps.driver_allocated_memory() / 1e9:.1f} GB")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"{trainable:,} / {total:,}")

    # Saving the results
    torch.save(model.state_dict(), "results/station1/resnet50_ft.pt")

    # Loading the model (for inference for example)
    model.load_state_dict(torch.load("results/station1/resnet50_ft.pt", map_location=device))