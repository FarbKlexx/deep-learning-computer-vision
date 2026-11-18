import timm
from timm.data.config import resolve_model_data_config
from timm.data.transforms_factory import create_transform
import torch

from src.data.augmentation import train_transform
from src.models.experiment import Experiment

MODEL_NAME = "vit_base_patch16_224"


class ViTLinearProbing(Experiment):
    def __init__(self, lr=1e-3):
        self.lr = lr
        self.name = f"vit_linprob_lr{lr:g}"

    def build_model(self, num_classes):
        vit = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
        self.data_cfg = resolve_model_data_config(vit)

        # gesamtes Backbone einfrieren
        for p in vit.parameters():
            p.requires_grad = False

        # nur Head trainieren
        for p in vit.head.parameters():
            p.requires_grad = True

        return vit

    def transforms(self):
        size = self.data_cfg["input_size"][-2:]
        train_tf = train_transform(size, self.data_cfg["mean"], self.data_cfg["std"])
        eval_tf = create_transform(**self.data_cfg, is_training=False)
        return train_tf, eval_tf

    def param_groups(self, model):
        # nur Head ist trainierbar
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": self.lr}]

    # nur Head speichern
    def save(self, model, out):
        torch.save(model.head.state_dict(), out / f"{self.name}_head.pt")
