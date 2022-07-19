from torch import nn
from .audio_encoder import SqueezeformerEncoder
from torch.nn import *
from .frontend import MelSpec

def get_layer(cfg):
    if cfg.name in nn.modules.__dict__:
        return nn.modules.__dict__[cfg.name](**cfg.params)
    return eval(cfg.name)(cfg.params)

class SpeechModel(nn.Module):
    """Some Information about SpeechModel"""
    def __init__(self,cfg):
        super(SpeechModel, self).__init__()
        self.frontend = get_layer(cfg.frontend)
        self.encoder  = get_layer(cfg.encoder)
        self.decoder  = get_layer(cfg.decoder)

    def forward(self, x,xn):
        xs,xn = self.frontend(x,xn)
        xs,xn = self.encoder(xs,xn)
        logits = self.decoder(xs.transpose(1,2))
        return logits,xn
