import torch.nn as nn
from torch import Tensor
from argparse import Namespace


_ACTIVATION_FUNCTIONS: nn.ModuleDict = nn.ModuleDict(
    {
        "relu": nn.ReLU,
        "prelu": nn.PReLU,
        "rrelu": nn.RReLU,
        "celu": nn.CELU
    }
)

class Block(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, cfg: Namespace) -> None:
        super(Block, self).__init__()
        self.layer = nn.ModuleList(nn.Linear(input_dim, output_dim))
        if cfg.batch_norm:
        self.layer = nn.Sequential(
            ,
            nn.BatchNorm1d(output_dim) if cfg.batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)

class ResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size, batch_norm=False):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            Block(input_size, input_size, hidden_size, batch_norm), nn.SiLU()
        )

    def forward(self, x):
        return x + self.block(x)


class ShortcutBlock(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, batch_norm=False):
        super(ShortcutBlock, self).__init__()
        self.block = nn.Sequential(
            Block(input_size, output_size, hidden_size, batch_norm), nn.SiLU()
        )
        self.shortcut = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)
