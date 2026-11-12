import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

from src.data.loading import load_raw, make_loaders
from src.training.evaluation import count_params, evaluate, plot_history
from src.training.trainer import Trainer
from src.utils.device import get_device

OUT = Path("results/station1")

# abstract template for a complete training-run independent of the model
class Experiment(ABC):
    name = "experiment"

    # model has to implement these itself
    @abstractmethod
    def build_model(self, num_classes) -> nn.Module: ...

    @abstractmethod
    def transforms(self) -> tuple[Callable, Callable]: ...  # (train_tf, eval_tf)

    @abstractmethod
    def param_groups(self, model) -> list[dict]: ... 

    # model can implement these itself if necessary
    def save(self, model):
        pass

    def run(self, epochs=10):
        # get which device we are training on for example:
        # cuda (for nvidia gpu), mps (apple gpu chip) or on cpu
        device = get_device()
        OUT.mkdir(parents=True, exist_ok=True) # create output dir

        # load data
        train_hf, val_hf, test_hf = load_raw()
        num_classes = train_hf.features["label"].num_classes

        # init the model
        model = self.build_model(num_classes).to(device)
        # apply transformation (augmentation for training and preprocessing for evaluation)
        train_tf, eval_tf = self.transforms()
        train_dl, val_dl, test_dl = make_loaders(train_hf, val_hf, test_hf, train_tf, eval_tf)
        
        # declare loss-function
        criterion = nn.CrossEntropyLoss()

        # init optimizer as AdamW (industry standard) for weight adjustment
        optimizer = torch.optim.AdamW(self.param_groups(model))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # output path
        best_path = OUT / f"{self.name}_best.pt"

        # trainingloop
        trainer = Trainer(model, optimizer, scheduler, criterion, device, best_path)
        history = trainer.fit(train_dl, val_dl, epochs)

        # test nur einmal am Ende, mit der besten Epoche
        model.load_state_dict(torch.load(best_path, map_location=device))
        test = evaluate(model, test_dl, criterion, device)

        self.save(model)

        result = {
            "name": self.name,
            "test_acc": test["acc"],
            "test_loss": test["loss"],
            "best_val_acc": max(h["val_acc"] for h in history),
            **count_params(model),
            "sec_per_epoch": sum(h["sec"] for h in history) / len(history),
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
        }
        (OUT / f"{self.name}.json").write_text(json.dumps(result, indent=2))
        (OUT / f"{self.name}_history.json").write_text(json.dumps(history))
        plot_history(history, OUT / f"{self.name}_curves.png")
        print(result)
        return result
