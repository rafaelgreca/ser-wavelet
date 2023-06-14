import torch
from src.data_augmentation import AudioAugment, SpecAugment, Denoiser
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
        data_augmentation_config: Dict,
        training: bool,
        data_augment_target: str
    ) -> None:
        self.X = X
        self.y = y
        self.feature_config = feature_config
        self.wavelet_config = wavelet_config
        self.data_augmentation_config = data_augmentation_config
        self.training = training
        self.data_augment_target = data_augment_target
        
    def __len__(self):
        return len(self.y)
    
    def _apply_augmentation_raw_audio(
        self,
        audio: torch.Tensor
    ) -> torch.Tensor:
        transformations = self.data_augmentation_config["techniques"]
        p = self.data_augmentation_config["p"]
        sample_rate = self.feature_config["sample_rate"]
        
        for transformation in transformations.keys():
            if transformation == "denoiser":
                augment = Denoiser(
                    filters=transformations[transformation]["filters"],
                    sample_rate=sample_rate,
                    p=p
                )
            elif transformation == "audioaugment":
                augment = AudioAugment(
                    transformations=transformations[transformation]["transformations"],
                    sample_rate=sample_rate,
                    p=p
                )
            
            audio = augment(audio)
        
        return audio
    
    def _apply_augmentation_feature(
        self,
        audio: torch.Tensor
    ) -> torch.Tensor:
        transformations = self.data_augmentation_config["techniques"]
        p = self.data_augmentation_config["p"]
        
        for transformation in transformations.keys():
            if transformation == "specaugment":
                augment = SpecAugment(
                    transformations=transformations[transformation]["transformations"],
                    p=p,
                    mask_samples=transformations[transformation]["mask_samples"]
                )
            
            audio = augment(audio)
        
        return audio
    
    def __getitem__(
        self,
        index: int
    ) -> Dict:
        batch = {}
        
        if self.y[index].argmax(dim=-1, keepdim=False).item() in self.data_augment_target and self.training and \
            self.data_augmentation_config["mode"] == "raw_audio":
            self._apply_augmentation_raw_audio(self.X[index, :, :])
        
        assert self.X[index, :, :].ndim == 2 and self.X[index, :, :].shape[0] == 1
        
        if self.feature_config["name"] == "melspectrogram":
            feat = extract_melspectrogram(
                audio=self.X[index, :, :],
                sample_rate=self.feature_config["sample_rate"],
                n_fft=self.feature_config["n_fft"],
                hop_length=self.feature_config["hop_length"],
                n_mels=self.feature_config["n_mels"]
            )
        elif self.feature_config["name"] == "mfcc":
            feat = extract_mfcc(
                audio=self.X[index, :, :],
                sample_rate=self.feature_config["sample_rate"],
                n_fft=self.feature_config["n_fft"],
                hop_length=self.feature_config["hop_length"],
                n_mfcc=self.feature_config["n_mfcc"]
            )
        
        assert feat.ndim == 3 and feat.shape[0] == 1
        
        if self.y[index].argmax(dim=-1, keepdim=False).item() in self.data_augment_target and self.training and \
            self.data_augmentation_config["mode"] == "feature":
            self._apply_augmentation_feature(feat)
        
        assert feat.ndim == 3 and feat.shape[0] == 1
        
        X, _ = extract_wavelet_from_spectrogram(
            spectrogram=feat.squeeze(0),
            wavelet=self.wavelet_config["name"],
            maxlevel=self.wavelet_config["level"],
            type=self.wavelet_config["type"],
            mode=self.wavelet_config["mode"]
        )
        
        assert X.ndim == 2
        
        batch["features"] = X.unsqueeze(0)
        batch["labels"] = self.y[index]
        return batch

def create_dataloader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    feature_config: Dict,
    wavelet_config: Dict,
    data_augmentation_config: Dict,
    data_augment_target: str,
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
        data_augmentation_config=data_augmentation_config,
        training=training,
        data_augment_target=data_augment_target
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