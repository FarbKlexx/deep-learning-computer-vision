import torch
from torchvision.transforms import v2


def train_transform(size, mean, std):
    return v2.Compose([
        v2.RandomResizedCrop(size),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])
