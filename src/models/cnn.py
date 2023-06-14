import torch
import torch.nn as nn
from src.models.utils import weight_init

class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()        
        self.input_channels = 1
        self.linear_input_features = 118784
            
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=64,
                kernel_size=(2, 2)
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(2, 2)
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(2, 2)
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(2, 2)
            ),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(2, 2)
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(2, 2)
            ),
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=(2, 2)
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(2, 2)
            ),
            nn.Flatten(),
            nn.Linear(
                in_features=self.linear_input_features,
                out_features=3
            )
        )
                    
        self.model.apply(weight_init)
    
    def forward(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        return self.model(X)