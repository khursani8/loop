import torch
from torch import nn
import timm

class TimmModel(nn.Module):
    def __init__(self, cfg):
        super(TimmModel, self).__init__()

        self.model = timm.create_model(cfg["arch"], pretrained=False, in_chans=3,num_classes=cfg["num_classes"])

    def forward(self, x):
        x = self.model(x)
        return x
