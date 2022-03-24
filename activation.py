import torch
from torch import nn
import torch.nn.functional as F


class ReLU_k(nn.Module):
    def __init__(self, p=1) -> None:
        super(ReLU_k, self).__init__()
        self.p = p

    def forward(self, x):
        return torch.pow(F.relu(x), self.p)


class Leaky_ReLU_k(nn.Module):
    def __init__(self, p=1, negative_slope=0.2) -> None:
        super(Leaky_ReLU_k, self).__init__()
        self.p = p
        self.negative_slope = negative_slope

    def forward(self, x):
        return torch.pow(F.leaky_relu(x, negative_slope=self.negative_slope), self.p)


# Fix Error in PyTorch nn.Hardswish Module
# From https://github.com/learnables/learn2learn/issues/212
# RuntimeError: derivative for hardswish_backward is not implemented
class Hardswish(nn.Module):  # export-friendly version of nn.Hardswish()
    @staticmethod
    def forward(x):
        # return x * F.hardsigmoid(x)  # for torchscript and CoreML
        return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0  # for torchscript, CoreML and ONNX

