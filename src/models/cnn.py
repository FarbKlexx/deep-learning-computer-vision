from torch import nn
from torchvision import models

from src.data.augmentation import train_transform
from src.models.experiment import Experiment


class CNNFinetune(Experiment): # inherts from Experiment class
    def __init__(self, lr=1e-4):
        self.weights = models.ResNet50_Weights.IMAGENET1K_V2
        self.lr = lr
        self.name = f"cnn_lr{self.lr:g}"

    def build_model(self, num_classes):
        model = models.resnet50(weights=self.weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # neuer Classifier Head
        return model

    def transforms(self):
        eval_tf = self.weights.transforms()
        train_tf = train_transform(eval_tf.crop_size, eval_tf.mean, eval_tf.std)
        return train_tf, eval_tf

    def param_groups(self, model):
        # ganzes Netz wird mit einer Lernrate trainiert
        return [{"params": list(model.parameters()), "lr": self.lr}]
