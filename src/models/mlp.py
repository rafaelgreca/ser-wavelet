import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(
        self
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(
                in_features=(4 + 1) * 3,
                out_features=64
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(
                in_features=64,
                out_features=128
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(
                in_features=128,
                out_features=256
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(
                in_features=256,
                out_features=3
            )
        )
    
    def forward(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        return self.model(X)