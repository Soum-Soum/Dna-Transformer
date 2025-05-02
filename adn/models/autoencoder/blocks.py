import torch.nn as nn


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None):
        super().__init__()
        layers = [nn.Linear(in_dim, out_dim)]
        if activation is not None:
            layers.append(activation)
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation=None):
        super().__init__()
        layers = [nn.Conv1d(in_channels, out_channels, kernel_size, padding="same")]
        if activation is not None:
            layers.append(activation)
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
