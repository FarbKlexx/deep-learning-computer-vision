import json
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from src.data.loading import load_raw, make_loaders
from src.training.evaluation import count_params, evaluate, plot_history
from src.training.trainer import Trainer
from src.utils.device import get_device

Transform = Callable[[Image.Image], torch.Tensor] # Datatype for transforms return statement


# abstract template for a complete training-run independent of the model
class Experiment(ABC):
    name = "experiment"  # used for the output file names

    # model has to implement these itself

    @abstractmethod
    def build_model(self, num_classes: int) -> nn.Module:
        """Returns the model, output is one logit per class."""

    @abstractmethod
    def transforms(self) -> tuple[Transform, Transform]:
        """Returns (train_tf, eval_tf), each turns one image into a tensor of the same size.

        Size and normalization have to match the pretrained weights. For train_tf use
        train_transform() so all models get the same augmentation.
        """

    @abstractmethod
    def param_groups(self, model: nn.Module) -> list[dict]:
        """Returns the trainable parameters, e.g. [{"params": [...], "lr": 1e-4}]."""

    # model can implement these itself if necessary
    def save(self, model: nn.Module, out: Path) -> None:
        """Saves the model additionally in its own format."""

    def run(self, out: str | Path, epochs: int = 10) -> dict:
        out = Path(out)
        # get which device we are training on for example:
        # cuda (for nvidia gpu), mps (apple gpu chip) or on cpu
        device = get_device()
        out.mkdir(parents=True, exist_ok=True) # create output dir

        # load data
        train_hf, val_hf, test_hf = load_raw()
        num_classes = train_hf.features["label"].num_classes

        # sanity check for dataset size
        n_train, n_val, n_test = len(train_hf), len(val_hf), len(test_hf)
        print("Dataset Size. n_train: ", n_train, "n_val: ", n_val, "n_test: ", n_test)

        # init the model
        model = self.build_model(num_classes).to(device)
        # apply transformation (augmentation for training and preprocessing for evaluation)
        train_tf, eval_tf = self.transforms()
        train_dl, val_dl, test_dl = make_loaders(train_hf, val_hf, test_hf, train_tf, eval_tf)
        
        # declare loss-function
        criterion = nn.CrossEntropyLoss()

        # sanity check for initial loss
        initial_loss = evaluate(model, val_dl, criterion, device)
        print("Intial Loss: ", initial_loss, "log(num_classes): ", math.log(num_classes))


        # init optimizer as AdamW (industry standard) for weight adjustment
        optimizer = torch.optim.AdamW(self.param_groups(model))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # output path
        best_path = out / f"{self.name}_best.pt"

        # trainingloop
        trainer = Trainer(model, optimizer, scheduler, criterion, device, best_path)
        history = trainer.fit(train_dl, val_dl, epochs)

        # test nur einmal am Ende, mit der besten Epoche
        model.load_state_dict(torch.load(best_path, map_location=device))
        test = evaluate(model, test_dl, criterion, device)

        self.save(model, out)

        result = {
            "name": self.name,
            "init_loss": initial_loss["loss"],
            "init_acc": initial_loss["acc"],
            "exp_init_loss": math.log(num_classes),
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test, 
            "test_acc": test["acc"],
            "test_loss": test["loss"],
            "best_val_acc": max(h["val_acc"] for h in history),
            **count_params(model),
            "sec_per_epoch": sum(h["sec"] for h in history) / len(history),
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
        }
        (out / f"{self.name}.json").write_text(json.dumps(result, indent=2))
        (out / f"{self.name}_history.json").write_text(json.dumps(history))
        plot_history(history, out / f"{self.name}_curves.png")
        print(result)
        return result
