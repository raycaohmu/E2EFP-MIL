import torch.nn as nn
from torch.jit.annotations import Dict

from collections import OrderedDict


class IntermediateLayerGetter(nn.ModuleDict):
    __annotations__ = {
            "return_layers": Dict[str, str]
    }

    def __init__(self, model, return_layers):
        ori_return_layers = return_layers.copy()
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = ori_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


if __name__ == "__main__":
    import torch
    from model import AttFPNMIL
    from torchvision.models import resnet34

    net = AttFPNMIL().feature_extractor
    #net = resnet34(pretrained=True)
    return_layers1 = {"4": "feat1", "5": "feat2", "6": "feat3", "7": "feat4"}

    x = torch.randn(2, 3, 256, 256)
    layergetter = IntermediateLayerGetter(net, return_layers1)
    out = layergetter(x)
