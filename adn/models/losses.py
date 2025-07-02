from torch import nn
import torch


class ArcFaceLayer(nn.Module):

    def __init__(
        self,
        dimension: int,
        n_classes: int,
        k_sub_centers: int,
        marging: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dimension = dimension
        self.n_classes = n_classes
        self.k_sub_centers = k_sub_centers
        self.marging = marging
        self.weight = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(n_classes, k_sub_centers, dimension))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        norm_x = nn.functional.normalize(x, dim=-1)
        norm_weight = nn.functional.normalize(self.weight, dim=-1)
        cosine_sims = torch.einsum(
            "bd, nkd -> bnk",
            norm_x,
            norm_weight,
        )
        polled_cosine_sims = cosine_sims.max(dim=-1).values
        
        theta = torch.acos(
            torch.clamp(polled_cosine_sims, -1.0, 1.0)
        )
        theta = theta + self.marging
        