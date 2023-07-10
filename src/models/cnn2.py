import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from src.models.utils import weight_init

class Extract_LSTM_Output(nn.Module):
    """
    Extracts only the output from the BiLSTM layer.
    """
    def forward(self, x):
        output, _ = x
        return output

class Attention_Layer(nn.Module):
    def __init__(
        self,
        n_feats: int
    ) -> None:
        super().__init__()
        self.w = nn.Linear(
            in_features=n_feats,
            out_features=n_feats
        )
    
    def forward(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        w = self.w(X)
        output = F.softmax(torch.mul(X, w), dim=1)
        return output
        
class FLB(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: Tuple
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                padding="valid"
            ),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU()
        )
    
    def forward(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        X = self.block(X)
        return X

class CNN2_Mode2(nn.Module):
    def __init__(
        self,
        num_classes: int
    ) -> None:
        super().__init__()
        self.in_channels = 5
        self.linear_input_features = 1103872
        
        self.cnn = nn.Sequential(
            FLB(
                input_channels=self.in_channels,
                output_channels=64,
                kernel_size=(2, 2)
            ),
            nn.MaxPool2d(kernel_size=(2, 2)),
            FLB(
                input_channels=64,
                output_channels=128,
                kernel_size=(2, 2)
            ),
            FLB(
                input_channels=128,
                output_channels=256,
                kernel_size=(2, 2)
            ),
            nn.MaxPool2d(kernel_size=(2, 2)),
            FLB(
                input_channels=256,
                output_channels=512,
                kernel_size=(2, 2)
            ),
            FLB(
                input_channels=512,
                output_channels=512,
                kernel_size=(2, 2)
            ),
            nn.Flatten(),
            nn.Linear(
                in_features=self.linear_input_features,
                out_features=128
            )
        )
        
        self.lstm = nn.Sequential(
            nn.LSTM(
                input_size=128,
                hidden_size=128,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            ),
            Extract_LSTM_Output()
        )
        
        self.model = nn.Sequential(
            self.cnn,
            self.lstm,
            Attention_Layer(256),
            nn.Linear(
                in_features=256,
                out_features=num_classes
            )
        )
        self.model.apply(weight_init)
    
    def forward(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        return self.model(X)
    
class CNN2_Mode1(nn.Module):
    def __init__(
        self,
        num_classes: int
    ) -> None:
        super().__init__()
        self.in_channels = 1
        self.linear_input_features = 1406464
        
        self.cnn = nn.Sequential(
            FLB(
                input_channels=self.in_channels,
                output_channels=64,
                kernel_size=(2, 2)
            ),
            nn.MaxPool2d(kernel_size=(2, 2)),
            FLB(
                input_channels=64,
                output_channels=128,
                kernel_size=(2, 2)
            ),
            FLB(
                input_channels=128,
                output_channels=256,
                kernel_size=(2, 2)
            ),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
            nn.Linear(
                in_features=self.linear_input_features,
                out_features=256
            )
        )
        
        self.model = nn.Sequential(
            self.cnn,
            nn.Linear(
                in_features=256,
                out_features=num_classes
            )
        )
        self.model.apply(weight_init)
    
    def forward(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        return self.model(X)