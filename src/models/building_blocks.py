from models.ModelBase import ModelBase
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import anndata as ad
from torch.utils.data import DataLoader


class Block(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, batch_norm=False):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size) if batch_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.block(x)


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
