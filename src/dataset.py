import numpy as np
import torch
from copy import deepcopy
from src.data_augmentation import AudioAugment, SpecAugment, Denoiser
from src.features import extract_melspectrogram, extract_mfcc, extract_wavelet_from_spectrogram, extract_wavelet_from_raw_audio
from src.utils import pad_features
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Union, Iterable

def _apply_augmentation_raw_audio(
    audio: torch.Tensor,
    data_augmentation_config: Dict,
    feature_config: Dict
) -> torch.Tensor:
    transformations = data_augmentation_config["techniques"]
    p = data_augmentation_config["p"]
    sample_rate = feature_config["sample_rate"]
    
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
    audio: torch.Tensor,
    data_augmentation_config: Dict,
    feature_config: Dict
) -> torch.Tensor:
    transformations = data_augmentation_config["techniques"]
    p = data_augmentation_config["p"]
    
    for transformation in transformations.keys():
        if transformation == "specaugment":
            augment = SpecAugment(
                transformations=transformations[transformation]["transformations"],
                p=p,
                mask_samples=transformations[transformation]["mask_samples"],
                feature=feature_config["name"]
            )
            
        audio = augment(audio)
    
    return audio

class Dataset_Mode2(Dataset):
    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        feature_config: Dict,
        wavelet_config: Dict,
        data_augmentation_config: Union[Dict, None],
        training: bool,
        data_augment_target: Union[str, None]
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
        
    def __getitem__(
        self,
        index: int
    ) -> Dict:
        batch = {}
        audio = deepcopy(self.X[index, :, :])
        
        if self.data_augment_target is not None:
            if self.y[index].argmax(dim=-1, keepdim=False).item() in self.data_augment_target and self.training and \
                self.data_augmentation_config["mode"] == "raw_audio":
                audio = _apply_augmentation_raw_audio(
                    audio=audio,
                    data_augmentation_config=self.data_augmentation_config,
                    feature_config=self.feature_config
                )
        
        assert audio.ndim == 2 and audio.shape[0] == 1
        
        coeffs = extract_wavelet_from_raw_audio(
            audio=audio.squeeze(0),
            wavelet=self.wavelet_config["name"],
            maxlevel=self.wavelet_config["level"],
            type=self.wavelet_config["type"],
            mode=self.wavelet_config["mode"]
        )
        
        assert len(coeffs) == self.wavelet_config["level"] + 1
        
        # transforming the coeffs to torch
        new_coeffs = [torch.from_numpy(coeffs[i]).unsqueeze(0) for i in range(len(coeffs))]

        # extracting the mel spectrogram from each wavelet coefficient       
        feats = []
        
        for coeff in new_coeffs:
            if self.feature_config["name"] == "mel_spectrogram":
                feat = extract_melspectrogram(
                    audio=coeff,
                    sample_rate=self.feature_config["sample_rate"],
                    n_fft=self.feature_config["n_fft"],
                    hop_length=self.feature_config["hop_length"],
                    n_mels=self.feature_config["n_mels"]
                )
            elif self.feature_config["name"] == "mfcc":
                feat = extract_mfcc(
                    audio=coeff,
                    sample_rate=self.feature_config["sample_rate"],
                    n_fft=self.feature_config["n_fft"],
                    hop_length=self.feature_config["hop_length"],
                    n_mfcc=self.feature_config["n_mfcc"]
                )
                                        
            if self.data_augment_target is not None:
                if self.y[index].argmax(dim=-1, keepdim=False).item() in self.data_augment_target and self.training and \
                    self.data_augmentation_config["mode"] == "feature":
                    feat = _apply_augmentation_feature(
                        audio=feat,
                        data_augmentation_config=self.data_augmentation_config,
                        feature_config=self.feature_config
                    )
                                        
            assert feat.ndim == 3 and feat.shape[0] == 1
            
            feats.append(feat)
        
        assert len(feats) == self.wavelet_config["level"] + 1
        
        # padding the mel spectrograms to be the same size
        max_height = max([x.size(1) for x in feats])
        max_width = max([x.size(2) for x in feats])

        feats = pad_features(
            features=feats,
            max_height=max_height,
            max_width=max_width
        )
        feats = torch.concat(feats, dim=0)
        feats = feats.permute(0, 2, 1) # time and frequency axis permutation
                  
        batch["features"] = feats
        batch["labels"] = self.y[index]
        return batch
    
class Dataset_Mode1(Dataset):
    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        feature_config: Dict,
        wavelet_config: Dict,
        data_augmentation_config: Union[Dict, None],
        training: bool,
        data_augment_target: Union[str, None]
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
        
    def __getitem__(
        self,
        index: int
    ) -> Dict:
        batch = {}
        audio = deepcopy(self.X[index, :, :])
        
        if self.data_augment_target is not None:
            if self.y[index].argmax(dim=-1, keepdim=False).item() in self.data_augment_target and self.training and \
                self.data_augmentation_config["mode"] == "raw_audio":
                audio = _apply_augmentation_raw_audio(
                    audio=audio,
                    data_augmentation_config=self.data_augmentation_config,
                    feature_config=self.feature_config
                )
        
        assert audio.ndim == 2 and audio.shape[0] == 1
                
        if self.feature_config["name"] == "mel_spectrogram":
            feat = extract_melspectrogram(
                audio=audio,
                sample_rate=self.feature_config["sample_rate"],
                n_fft=self.feature_config["n_fft"],
                hop_length=self.feature_config["hop_length"],
                n_mels=self.feature_config["n_mels"]
            )
        elif self.feature_config["name"] == "mfcc":
            feat = extract_mfcc(
                audio=audio,
                sample_rate=self.feature_config["sample_rate"],
                n_fft=self.feature_config["n_fft"],
                hop_length=self.feature_config["hop_length"],
                n_mfcc=self.feature_config["n_mfcc"]
            )
        
        assert feat.ndim == 3 and feat.shape[0] == 1

        if self.data_augment_target is not None:
            if self.y[index].argmax(dim=-1, keepdim=False).item() in self.data_augment_target and self.training and \
                self.data_augmentation_config["mode"] == "feature":
                feat = _apply_augmentation_feature(
                    audio=feat,
                    data_augmentation_config=self.data_augmentation_config,
                    feature_config=self.feature_config
                )
        
        assert feat.ndim == 3 and feat.shape[0] == 1
        
        feat = feat.permute(0, 2, 1) # time and frequency axis permutation
        
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
    data_augmentation_config: Union[Dict, None],
    data_augment_target: Union[str, None],
    mode: str,
    worker_init_fn: Iterable,
    generator: torch.Generator,
    num_workers: int = 0,
    shuffle: bool = True,
    training: bool = True
) -> DataLoader:
    
    # creating the custom dataset
    if mode == "mode_1":
        dataset = Dataset_Mode1(
            X=X,
            y=y,
            feature_config=feature_config,
            wavelet_config=wavelet_config,
            data_augmentation_config=data_augmentation_config,
            training=training,
            data_augment_target=data_augment_target
        )
    elif mode == "mode_2":
        dataset = Dataset_Mode2(
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
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=generator
    )
    
    return dataloader