from typing import Any
import torch
import torch.nn as nn
from src.models.utils import weight_init

class CNN_Mode2_Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
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
            nn.Flatten()
        )

        self.block.apply(weight_init)
        
    def forward(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        return self.block(X)
    
class CNN_Mode2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.models = nn.ModuleList(
            [CNN_Mode2_Block() for _ in range(5)]
        ) 
        
        self.output_layer = nn.Linear(
            in_features=555520,
            out_features=3
        )
        
        self.output_layer.apply(weight_init)
    
    def forward(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        outputs = []
        
        for i, _ in enumerate(self.models):
            output = self.models[i](X[:, i, :, :].unsqueeze(1)).unsqueeze(1)
            outputs.append(output)
            
        outputs = torch.concat(outputs, dim=1)
        outputs = torch.flatten(outputs, start_dim=1, end_dim=-1)
        output = self.output_layer(outputs)
        return output
    
class CNN_Mode1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_channels = 1
        self.linear_input_features = 276480
            
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