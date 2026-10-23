import torch
from datasets import load_dataset
from torch.utils.data import DataLoader


class _ApplyTransform:
    def __init__(self, tf):
        self.tf = tf

    def __call__(self, batch):
        batch["image"] = [self.tf(img.convert("RGB")) for img in batch["image"]]
        return batch


def _collate(batch):
    return torch.stack([b["image"] for b in batch]), torch.tensor([b["label"] for b in batch])


def make_loaders(train_tf, eval_tf, batch_size=32, eval_batch_size=64, num_workers=4, seed=67):
    """Load EuroSAT and return DataLoaders for train/val/test.

    The dataset ships a train and a test split. 10 % of the train split
    is held out as validation set; the test split stays untouched.

    Returns: train_dl, val_dl, test_dl, class_names
    """
    ds = load_dataset("tanganke/eurosat")
    split = ds["train"].train_test_split(test_size=0.1, stratify_by_column="label", seed=seed)
    train_hf, val_hf, test_hf = split["train"], split["test"], ds["test"]  # 19.440 / 2.160 / 2.700
    class_names = train_hf.features["label"].names

    def loader(hf, tf, bs, shuffle):
        return DataLoader(hf.with_transform(_ApplyTransform(tf)), batch_size=bs,
                          shuffle=shuffle, num_workers=num_workers, collate_fn=_collate)

    train_dl = loader(train_hf, train_tf, batch_size, True)
    val_dl = loader(val_hf, eval_tf, eval_batch_size, False)
    test_dl = loader(test_hf, eval_tf, eval_batch_size, False)
    return train_dl, val_dl, test_dl, class_names
