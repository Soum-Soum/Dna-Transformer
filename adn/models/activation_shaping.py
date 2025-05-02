import torch


class ActivationShapingS(torch.nn.Module):

    def __init__(self, pruning_level: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pruning_level = pruning_level

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply activation type S to the inputs.
        for details, see: https://arxiv.org/pdf/2209.09858

        Args:
            inputs (torch.Tensor): _input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            torch.Tensor: output tensor of the same shape as inputs
        """
        batch_size, seq_len, hidden_size = inputs.shape
        fattened = inputs.reshape((batch_size, -1))
        sum_1 = torch.sum(fattened, dim=1)

        percentile = (
            torch.quantile(fattened, self.pruning_level, dim=-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, seq_len, hidden_size)
        )
        x_with_zeros = torch.where(
            inputs > percentile, inputs, torch.zeros_like(inputs)
        )
        sum_2 = torch.sum(x_with_zeros.reshape((batch_size, -1)), dim=1)

        # Avoid division by zero
        exp_ratio = (
            torch.exp(sum_1 / (sum_2 + 1e-6))
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, seq_len, hidden_size)
        )

        return x_with_zeros * exp_ratio
