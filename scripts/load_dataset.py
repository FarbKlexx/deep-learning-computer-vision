import json
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

# Train-/Val-Indizes, die notebooks/dataset_analysis.ipynb als Duplikate markiert
DROP_FILE = Path(__file__).with_name("drop_indices.json")


class _ApplyTransform:
    def __init__(self, tf):
        self.tf = tf

    def __call__(self, batch):
        batch["image"] = [self.tf(img.convert("RGB")) for img in batch["image"]]
        return batch

def load_raw(name='tanganke/sun397', seed=67):
    ds = load_dataset(name)
    split = ds["train"].train_test_split(test_size=0.1, stratify_by_column="label", seed=seed)
    return split["train"], split["test"], ds["test"]

def _drop(hf, idx):
    idx = set(idx)
    return hf.select([i for i in range(len(hf)) if i not in idx])


def _collate(batch):
    return torch.stack([b["image"] for b in batch]), torch.tensor([b["label"] for b in batch])


def make_loaders(train_tf, eval_tf, batch_size=32, eval_batch_size=64, num_workers=4, seed=67):
    """Load Stanford Cars and return DataLoaders for train/val/test.

    The dataset ships a train and a test split. 10 % of the train split
    is held out as validation set; the test split stays untouched.
    Duplicates listed in DROP_FILE are removed from train and val
    (the indices are only valid for the seed they were computed with).

    Returns: train_dl, val_dl, test_dl, class_names
    """
    train_hf, val_hf, test_hf = load_raw(seed=seed)
    class_names = train_hf.features["label"].names

    if DROP_FILE.exists():
        drop = json.loads(DROP_FILE.read_text())
        if drop.get("seed", seed) != seed:
            raise ValueError(f"{DROP_FILE.name} wurde mit seed={drop['seed']} erstellt, nicht {seed}")
        train_hf = _drop(train_hf, drop["train"])
        val_hf = _drop(val_hf, drop["val"])

    def loader(hf, tf, bs, shuffle):
        return DataLoader(hf.with_transform(_ApplyTransform(tf)), batch_size=bs,
                          shuffle=shuffle, num_workers=num_workers, collate_fn=_collate)

    train_dl = loader(train_hf, train_tf, batch_size, True)
    val_dl = loader(val_hf, eval_tf, eval_batch_size, False)
    test_dl = loader(test_hf, eval_tf, eval_batch_size, False)
    return train_dl, val_dl, test_dl, class_names
