import timm
from peft import LoraConfig, get_peft_model

from src.data.augmentation import train_transform
from src.models.experiment import OUT, Experiment

MODEL_NAME = "vit_base_patch16_224.orig_in21k_ft_in1k"


class ViTLoRA(Experiment):
    def __init__(self, rank=8, targets=("attn.qkv", "attn.proj"), lr=1e-3):
        self.rank = rank
        self.targets = list(targets)
        self.lr = lr
        self.name = f"vit_lora_r{rank}_{'-'.join(self.targets)}"

    def build_model(self, num_classes):
        vit = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
        self.data_cfg = timm.data.resolve_model_data_config(vit)
        config = LoraConfig(r=self.rank, lora_alpha=self.rank, target_modules=self.targets,
                            modules_to_save=["head"])
        model = get_peft_model(vit, config)  # Basismodell bleibt eingefroren
        model.print_trainable_parameters()
        return model

    def transforms(self):
        size = self.data_cfg["input_size"][-2:]
        train_tf = train_transform(size, self.data_cfg["mean"], self.data_cfg["std"])
        eval_tf = timm.data.create_transform(**self.data_cfg, is_training=False)
        return train_tf, eval_tf

    def param_groups(self, model):
        # nur LoRA-Matrizen und Head sind trainierbar
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": self.lr}]

    def save(self, model):
        model.save_pretrained(OUT / self.name)
