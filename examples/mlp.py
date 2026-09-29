"""KernelOpt fixture: small MLP (KernelBench `get_model`/`get_inputs` convention).

Used by the walking-skeleton E2E run: `kernelopt run examples/mlp.py --provider mock`.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features=256, hidden=512, out_features=128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def get_model():
    return Model()


def get_inputs():
    return [torch.randn(64, 256, device="cuda")]


if __name__ == "__main__":
    m = get_model().cuda()
    print(m(*get_inputs()).shape)
