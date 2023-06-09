import torch
import torchaudio
import torch.nn.functional as F
import pandas as pd
from src.utils import one_hot_encoder
from typing import Tuple, List, Union

def normalize(
    samples: torch.Tensor
) -> torch.Tensor:
    """
    Normalize the audio's samples.

    Args:
        samples (torch.Tensor): the audio's samples.

    Returns:
        torch.Tensor: the normalized audio's samples (values between -1 and 1).
    """
    # expected samples shape is (1, total_samples)
    reshaped_samples = samples.squeeze(0)
    
    # getting the maximum absolute amplitude
    max_amplitude = torch.max(torch.abs(reshaped_samples))
    
    # normalizing the samples and getting back to the original shape
    samples = reshaped_samples / max_amplitude
    samples = samples.unsqueeze(0)
    
    return samples

def stereo_to_mono(
    audio: torch.Tensor
) -> torch.Tensor:
    """
    Converts a stereo audio to mono.
    
    Args:
        audio (torch.Tensor): the audio's waveform (stereo).
    
    Returns:
        torch.Tensor: the audio's waveform (mono).
    """
    audio = torch.mean(audio, dim=0, keepdim=True)
    return audio

def resample_audio(
    audio: torch.Tensor,
    sample_rate: int,
    new_sample_rate: int
) -> torch.Tensor:
    """
    Resamples a given audio.

    Args:
        audio (torch.Tensor): the audio's waveform.
        sample_rate (int): the original audio's sample rate.
        new_sample_rate (int): the new audio's sample rate.

    Returns:
        torch.Tensor: the resampled audio's waveform.
    """
    transform = torchaudio.transforms.Resample(
        orig_freq=sample_rate,
        new_freq=new_sample_rate
    )
    audio = transform(audio)
    return audio

def read_audio(
    path: str,
    to_mono: bool = True,
    sample_rate: int = 16000
) -> Tuple[torch.Tensor, int]:
    """
    Reads a audio file.

    Args:
        path (str): the audio file's path.
        to_mono (bool, optional): convert the signal to mono. Defaults to True.
        sample_rate (int, optional): resample the audio to that specific sample rate.
                                     Defaults to 16000.

    Returns:
        Tuple[torch.Tensor, int]: the audio waveform and the sample rate.
    """
    audio, sr = torchaudio.load(filepath=path, normalize=False)
    
    # resampling the audio to that specific sample rate (if necessary)
    if sample_rate != sr:
        audio = resample_audio(
            audio=audio,
            sample_rate=sr,
            new_sample_rate=sample_rate
        )
        sr = sample_rate
    
    # converting to mono (if necessary)
    if to_mono and audio.shape[0] > 1:
        audio = stereo_to_mono(audio=audio)
    
    return audio, sr

def pad_data(
    features: List,
    max_frames: int
) -> torch.Tensor:
    """
    Auxiliary function to pad the features.
    
    Args:
        features (List): the features that will be padded (mfcc, spectogram or mel_spectogram).
        max_frames (int): the max frames value.
    
    Returns:
        List: the padded features.
    """
    features = [
        F.pad(f, (0, max_frames - f.size(1)))
        for f in features
    ]
    return features

def processing(
    df: pd.DataFrame,
    to_mono: bool,
    sample_rate: int,
    apply_one_hot_encoder: bool = True,
    max_frames: Union[int, None] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Function responsible for the the processing step.
    It reads the audios (converts to mono and resamples, if necessary),
    normalizes, pads and one hot encode the labels (if necessary).

    Args:
        df (pd.DataFrame): the audios dataframe.
        to_mono (bool): if the audios should be converted to mono.
        sample_rate (int): the audios new sample rate.
        apply_one_hot_encoder (bool, optional): if the one hot encoding
                                                should be applied. Defaults to True.
        max_frames (int): the maximum number of samples.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: the audios samples and labels.
    """
    data = []
    labels = []
    max_frames = 0 if max_frames is None else max_frames
    
    for label, file_path in zip(df["label"], df["file"]):
        # reading the audio
        audio, sr = read_audio(
            path=file_path,
            to_mono=to_mono,
            sample_rate=sample_rate
        )
        
        assert sr == sample_rate
        
        # normalizing the audio's samples
        audio = normalize(audio)

        assert audio.max().round().item() <= 1.0 and audio.min().round().item() >= -1.0
        
        if audio.shape[1] > max_frames:
            max_frames = audio.shape[1]
        
        data.append(audio)
        labels.append(label)
    
    # padding the audio's data
    data = pad_data(
        features=data,
        max_frames=max_frames
    )
    
    data = torch.cat(data, 0).to(dtype=torch.float32)
    data = data.unsqueeze(1)
    labels = torch.as_tensor(labels, dtype=torch.long)
    
    # applying one hot encoding to the labels
    if apply_one_hot_encoder:
        labels = one_hot_encoder(labels)
        
    return data, labels