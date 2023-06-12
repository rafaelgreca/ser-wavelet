import torch
from src.features import extract_melspectrogram, extract_mfcc, extract_wavelet_from_spectrogram
from torch.utils.data import Dataset, DataLoader
from typing import Dict

class Dataset(Dataset):
    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        feature_config: Dict,
        wavelet_config: Dict,
        training: bool
    ) -> None:
        self.X = X
        self.y = y
        self.feature_config = feature_config
        self.wavelet_config = wavelet_config
        self.training = training
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(
        self,
        index: int
    ) -> Dict:
        batch = {}
        
        if self.feature_config["name"] == "melspectrogram":
            feat = extract_melspectrogram(
                audio=self.X[index, :, :].squeeze(0),
                sample_rate=self.feature_config["sample_rate"],
                n_fft=self.feature_config["n_fft"],
                hop_length=self.feature_config["hop_length"],
                n_mels=self.feature_config["n_mels"]
            )
        elif self.feature_config["name"] == "mfcc":
            feat = extract_mfcc(
                audio=self.X[index, :, :].squeeze(0),
                sample_rate=self.feature_config["sample_rate"],
                n_fft=self.feature_config["n_fft"],
                hop_length=self.feature_config["hop_length"],
                n_mfcc=self.feature_config["n_mfcc"]
            )
            
        X, _ = extract_wavelet_from_spectrogram(
            spectrogram=feat,
            wavelet=self.wavelet_config["name"],
            maxlevel=self.wavelet_config["level"],
            type=self.wavelet_config["type"],
            mode=self.wavelet_config["mode"]
        )
                
        batch["features"] = X.unsqueeze(0)
        batch["labels"] = self.y[index]
        return batch

def create_dataloader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    feature_config: Dict,
    wavelet_config: Dict,
    num_workers: int = 0,
    shuffle: bool = True,
    training: bool = True
) -> DataLoader:
    # creating the custom dataset
    dataset = Dataset(
        X=X,
        y=y,
        feature_config=feature_config,
        wavelet_config=wavelet_config,
        training=training
    )
    
    # creating the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=True
    )
    
    return dataloader