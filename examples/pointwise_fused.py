"""KernelOpt fixture: pointwise chain Inductor typically splits into multiple kernels."""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n=4096):
        super().__init__()
        self.n = n

    def forward(self, x):
        x = torch.nn.functional.gelu(x)
        x = x * 1.5 + 0.3
        x = torch.clamp(x, -1.0, 1.0)
        return x


def get_model():
    return Model()


def get_inputs():
    return [torch.randn(256, 4096, device="cuda")]


if __name__ == "__main__":
    m = get_model().cuda()
    print(m(*get_inputs()).shape)
