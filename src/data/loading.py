import json
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

DATASET = "tanganke/sun397"
SEED = 67
DROP_FILE = Path(__file__).resolve().parents[2] / "results/station1/drop.json" # load dropfile from notebook data analysis

# load raw data and split into train, val and test sets
def load_raw(clean=True):
    # load and split
    ds = load_dataset(DATASET)
    split = ds["train"].train_test_split(test_size=0.1, stratify_by_column="label", seed=SEED)
    train, val, test = split["train"], split["test"], ds["test"]

    # remove unclean data
    if clean and DROP_FILE.exists():
        drop = json.loads(DROP_FILE.read_text())
        if drop.get("seed", SEED) != SEED:
            raise ValueError(f"drop.json passt nicht zum Seed {SEED}")
        drop_train, drop_val = set(drop["train"]), set(drop["val"])
        train = train.select([i for i in range(len(train)) if i not in drop_train])
        val = val.select([i for i in range(len(val)) if i not in drop_val])

    return train, val, test

# data-augmentation on the fly, adapter between Hugging Face and torchvision
class ApplyTransform:
    # constructor when creating the object
    def __init__(self, tf):
        self.tf = tf

    # call the object
    def __call__(self, batch):
        batch["image"] = [self.tf(img.convert("RGB")) for img in batch["image"]]
        return batch

# input: list of datapoints in one batch
# output: final batch format
# format:
# (images,                  labels)
# (Tensor[32,3,224,224],    Tensor[32])
def collate(datapoints):
    return torch.stack([b["image"] for b in datapoints]), torch.tensor([b["label"] for b in datapoints])

# for batchwise data loading while training
def make_loaders(train_hf, val_hf, test_hf, train_tf, eval_tf, num_workers=8):
    train_dl = DataLoader(train_hf.with_transform(ApplyTransform(train_tf)), batch_size=32,
                          shuffle=True, num_workers=num_workers, collate_fn=collate)
    # Val und Test ohne Augmentierung, nur die gleiche Vorverarbeitung
    val_dl = DataLoader(val_hf.with_transform(ApplyTransform(eval_tf)), batch_size=64,
                        shuffle=False, num_workers=num_workers, collate_fn=collate)
    test_dl = DataLoader(test_hf.with_transform(ApplyTransform(eval_tf)), batch_size=64,
                         shuffle=False, num_workers=num_workers, collate_fn=collate)
    return train_dl, val_dl, test_dl
