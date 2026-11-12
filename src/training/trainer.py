import time

import torch

from src.training.evaluation import evaluate


class Trainer:
    # constructor
    def __init__(self, model, optimizer, scheduler, criterion, device, best_path):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.best_path = best_path

    
    def train_one_epoch(self, loader):
        self.model.train()
        total = 0.0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(images), labels)
            loss.backward()
            self.optimizer.step()
            total += loss.item() * images.size(0)
        return total / len(loader.dataset)

    def fit(self, train_dl, val_dl, epochs):
        history, best_acc = [], 0.0
        for epoch in range(epochs):
            start = time.perf_counter()
            train_loss = self.train_one_epoch(train_dl)
            sec = time.perf_counter() - start
            val = evaluate(self.model, val_dl, self.criterion, self.device)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"],
                            "val_acc": val["acc"], "lr": self.scheduler.get_last_lr()[0], "sec": sec})

            # Early Stopping
            if val["acc"] > best_acc:
                best_acc = val["acc"]
                torch.save(self.model.state_dict(), self.best_path)

            self.scheduler.step()
            print(f"epoch {epoch}  train_loss {train_loss:.3f}  val_loss {val['loss']:.3f}  "
                  f"val_acc {val['acc']:.3f}  {sec:.0f}s", flush=True)
        return history
