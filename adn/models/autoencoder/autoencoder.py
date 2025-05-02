import torch
import torch.nn as nn
import pytorch_lightning as pl
from pydantic import BaseModel
from typing import Literal, List, Optional, Callable, Any
from .blocks import LinearBlock, Conv1dBlock


class AutoencoderConfig(BaseModel):
    """
    Configuration for the BinarySNPsAutoencoder model.
    Allows full parametrization of the architecture and training hyperparameters.
    """

    sequence_len: int
    latent_dim: int = 16
    encoder_layers: int = 3
    decoder_layers: int = 3
    hidden_dim: int = 128
    layer_type: Literal["linear", "conv1d"] = "linear"
    activation: Literal["relu", "gelu"] = "relu"
    kernel_size: int = 3  # for conv1d
    lr: float = 1e-3
    input_channels: int = 2


def get_activation(name: str) -> nn.Module:
    """
    Returns the activation function module given its name.
    """
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation: {name}")


def _make_layers(
    n_layers: int,
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    block_cls: Callable[..., nn.Module],
    activation: nn.Module,
    is_encoder: bool = True,
) -> nn.Sequential:
    """
    Utility to build a stack of blocks (Linear or Conv1d) for encoder/decoder.
    """
    layers = []
    for i in range(n_layers):
        this_in = in_dim if i == 0 else hidden_dim
        this_out = out_dim if (i == n_layers - 1) else hidden_dim
        act = activation if i < n_layers - 1 else None
        layers.append(block_cls(this_in, this_out, act))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """
    Encoder module for the autoencoder. Supports linear or conv1d layers.
    """

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        act = get_activation(config.activation)
        if config.layer_type == "linear":
            self.encoder = _make_layers(
                config.encoder_layers,
                config.sequence_len * config.input_channels,
                config.latent_dim,
                config.hidden_dim,
                LinearBlock,
                act,
                is_encoder=True,
            )
        else:
            self.encoder = _make_layers(
                config.encoder_layers,
                config.input_channels,
                config.latent_dim,
                config.hidden_dim,
                lambda in_c, out_c, act: Conv1dBlock(
                    in_c, out_c, config.kernel_size, act
                ),
                act,
                is_encoder=True,
            )
            self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the encoder.
        Args:
            x: Input tensor of shape (B, SequenceLen, 2)
        Returns:
            Latent representation tensor of shape (B, latent_dim)
        """
        if self.config.layer_type == "linear":
            x = x.view(x.size(0), -1)
            z = self.encoder(x)
        else:
            x = x.permute(0, 2, 1)
            z = self.encoder(x)
            z = self.final_pool(z).squeeze(-1)
        return z


class Decoder(nn.Module):
    """
    Decoder module for the autoencoder. Supports linear or conv1d layers.
    """

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        act = get_activation(config.activation)
        if config.layer_type == "linear":
            self.decoder = _make_layers(
                config.decoder_layers,
                config.latent_dim,
                config.sequence_len * config.input_channels,
                config.hidden_dim,
                LinearBlock,
                act,
                is_encoder=False,
            )
            self.decoder = nn.Sequential(self.decoder, nn.Sigmoid())
        else:
            self.decoder = _make_layers(
                config.decoder_layers,
                config.latent_dim,
                config.input_channels,
                config.hidden_dim,
                lambda in_c, out_c, act: Conv1dBlock(
                    in_c, out_c, config.kernel_size, act
                ),
                act,
                is_encoder=False,
            )
            self.decoder = nn.Sequential(self.decoder, nn.Sigmoid())
        self.config = config

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the decoder.
        Args:
            z: Latent tensor of shape (B, latent_dim)
        Returns:
            Reconstructed tensor of shape (B, SequenceLen, 2)
        """
        if self.config.layer_type == "linear":
            x_hat = self.decoder(z)
            x_hat = x_hat.view(
                z.size(0), self.config.sequence_len, self.config.input_channels
            )
        else:
            z = z.unsqueeze(-1).repeat(1, 1, self.config.sequence_len)
            x_hat = self.decoder(z)
            x_hat = x_hat.permute(0, 2, 1)  # (B, SeqLen, 2)
        return x_hat


class BinarySNPsAutoencoder(pl.LightningModule):
    """
    PyTorch Lightning module for a configurable autoencoder on binary SNP data.
    """

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.save_hyperparameters(config.dict())
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.loss_fn = nn.BCELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        Args:
            x: Input tensor of shape (B, SequenceLen, 2)
        Returns:
            Reconstructed tensor of shape (B, SequenceLen, 2)
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Training step for the autoencoder.
        """
        x = batch
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Validation step for the autoencoder.
        """
        x = batch
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Returns the optimizer for training.
        """
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)
