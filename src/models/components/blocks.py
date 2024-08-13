from argparse import Namespace

import torch.nn as nn
from torch import Tensor

_ACTIVATION_FUNCTIONS: nn.ModuleDict = nn.ModuleDict(
    {"relu": nn.ReLU, "prelu": nn.PReLU, "rrelu": nn.RReLU, "celu": nn.CELU}
)


class Block(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        batch_norm: bool,
        dropout: float = 0.0,
        activation=nn.ReLU(),
    ) -> None:
        super(Block, self).__init__()

        if batch_norm:
            self.layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                activation,
                nn.Linear(hidden_dim, output_dim),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            )
        else:
            self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        activation=nn.ReLU(),
        batch_norm=False,
        dropout: float = 0.0,
    ):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            Block(input_size, input_size, hidden_size, batch_norm, dropout, activation),
            activation,
        )

    def forward(self, x):
        return x + self.block(x)


class ShortcutBlock(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        activation=nn.ReLU(),
        batch_norm=False,
        dropout: float = 0.0,
    ):
        super(ShortcutBlock, self).__init__()

        self.block = nn.Sequential(
            Block(
                input_size, hidden_size, output_size, batch_norm, dropout, activation
            ),
            activation,
        )
        self.shortcut = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(
                cfg.first_modality_dim + cfg.second_modality_dim, cfg.encoder_hidden_dim
            ),
            nn.BatchNorm1d(cfg.encoder_hidden_dim) if cfg.batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout_rate),
            nn.Linear(cfg.encoder_hidden_dim, cfg.encoder_hidden_dim),
            nn.BatchNorm1d(cfg.encoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout_rate),
            nn.Linear(cfg.encoder_hidden_dim, cfg.encoder_hidden_dim),
            nn.BatchNorm1d(cfg.encoder_hidden_dim) if cfg.batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout_rate),
            nn.Linear(cfg.encoder_hidden_dim, cfg.encoder_out_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.decoder_hidden_dim),
            nn.BatchNorm1d(cfg.decoder_hidden_dim) if cfg.batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout_rate),
            nn.Linear(cfg.decoder_hidden_dim, cfg.decoder_hidden_dim),
            nn.BatchNorm1d(cfg.decoder_hidden_dim) if cfg.batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout_rate),
            nn.Linear(cfg.decoder_hidden_dim, cfg.decoder_hidden_dim),
            nn.BatchNorm1d(cfg.decoder_hidden_dim) if cfg.batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout_rate),
            nn.Linear(
                cfg.decoder_hidden_dim, cfg.first_modality_dim + cfg.second_modality_dim
            ),
        )

    def forward(self, z):
        decoded = self.decoder(z)
        return decoded
