import torch
import torchaudio
import torch.nn.functional as F
import pandas as pd
import os
from torch.nn.functional import one_hot
from sklearn.model_selection import StratifiedKFold, train_test_split
from typing import Tuple, List

# making sure the experiments are reproducible
seed = 2109


def one_hot_encoder(labels: torch.Tensor, num_classes: int = -1) -> torch.Tensor:
    """
    Encode the labels into the one hot format.

    Args:
        labels (torch.Tensor): the data labels.
        num_classes (int): the number of classes present in the data.

    Returns:
        torch.Tensor: the data labels one hot encoded.
    """
    return one_hot(labels, num_classes=num_classes)


def save(path: str, name: str, tensor: torch.Tensor) -> None:
    """
    Saves a PyTorch tensor.

    Args:
        path (str): The output path.
        name (str): The file name.
        tensor (torch.Tensor): The tensor which will be saved.
    """
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, f"{name}.pth")
    torch.save(tensor, path)


def split_data(
    X: torch.Tensor,
    y: torch.Tensor,
    dataset: str,
    output_path: str,
    k_fold: int = 0,
    apply_one_hot_encoder: bool = True,
) -> None:
    """
    Split the training data into training and validation set.
    If the dataset is 'coraa', then the test set will be created using
    the official test data.

    Args:
        X (torch.Tensor): the features tensor.
        y (torch.Tensor): the labels tensor.
        dataset (str): which dataset is being used.
        output_path (str): the output path where the data will be saved.
        k_fold (int, optional): how many folds the data will be spliited into.
                                If zero, then normal split will be applied.
                                Otherwise, kfold will be applied. Defaults to 0.
        apply_one_hot_encoder (bool, optional): if the one hot encoder will be applied. Defaults to True.
    """
    skf = None
    if dataset == "coraa":
        num_classes = 3
    elif dataset == "emodb":
        num_classes = 7
    elif dataset == "ravdess":
        num_classes = 8
    elif dataset == "savee":
        num_classes = 7

    if not k_fold is None:
        skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=seed)

        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train = X[train_index, :, :]
            y_train = y[train_index]

            X_valid = X[test_index, :, :]
            y_valid = y[test_index]

            if apply_one_hot_encoder:
                y_train = one_hot_encoder(labels=y_train, num_classes=num_classes)
                y_valid = one_hot_encoder(labels=y_valid, num_classes=num_classes)

            folder_path = os.path.join(output_path, dataset, f"fold{i}")

            save(path=folder_path, name="X_train", tensor=X_train)
            save(path=folder_path, name="y_train", tensor=y_train)

            save(path=folder_path, name="X_valid", tensor=X_valid)
            save(path=folder_path, name="y_valid", tensor=y_valid)

    else:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, stratify=y, shuffle=True, random_state=seed
        )

        if apply_one_hot_encoder:
            y_train = one_hot_encoder(labels=y_train, num_classes=num_classes)
            y_valid = one_hot_encoder(labels=y_valid, num_classes=num_classes)

        folder_path = os.path.join(output_path, dataset)

        save(path=folder_path, name="X_train", tensor=X_train)
        save(path=folder_path, name="y_train", tensor=y_train)

        save(path=folder_path, name="X_valid", tensor=X_valid)
        save(path=folder_path, name="y_valid", tensor=y_valid)


def normalize(samples: torch.Tensor) -> torch.Tensor:
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


def stereo_to_mono(audio: torch.Tensor) -> torch.Tensor:
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
    audio: torch.Tensor, sample_rate: int, new_sample_rate: int
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
        orig_freq=sample_rate, new_freq=new_sample_rate
    )
    audio = transform(audio)
    return audio


def read_audio(
    path: str, to_mono: bool = True, sample_rate: int = 16000
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
    audio, sr = torchaudio.load(filepath=path)

    # resampling the audio to that specific sample rate (if necessary)
    if sample_rate != sr:
        audio = resample_audio(audio=audio, sample_rate=sr, new_sample_rate=sample_rate)
        sr = sample_rate

    # converting to mono (if necessary)
    if to_mono and audio.shape[0] > 1:
        audio = stereo_to_mono(audio=audio)

    return audio, sr


def pad_data(features: List, max_frames: int) -> torch.Tensor:
    """
    Auxiliary function to pad the features.

    Args:
        features (List): the features that will be padded (mfcc, spectogram or mel_spectogram).
        max_frames (int): the max frames value.

    Returns:
        List: the padded features.
    """
    features = [F.pad(f, (0, max_frames - f.size(1))) for f in features]
    return features


def processing(
    df: pd.DataFrame, to_mono: bool, sample_rate: int, max_samples: int
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Function responsible for the the processing step.
    It reads the audios (converts to mono and resamples, if necessary),
    normalizes, pads and one hot encode the labels (if necessary).

    Args:
        df (pd.DataFrame): the audios dataframe.
        to_mono (bool): if the audios should be converted to mono.
        sample_rate (int): the audios new sample rate.
        max_samples (int): the maximum samples value.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: the audios samples, labels
    """
    data = []
    labels = []

    for label, file_path in zip(df["label"], df["file"]):
        # reading the audio
        audio, sr = read_audio(path=file_path, to_mono=to_mono, sample_rate=sample_rate)

        assert sr == sample_rate

        # normalizing the audio's samples
        # audio = normalize(audio)

        assert audio.max().round().item() <= 1.0 and audio.min().round().item() >= -1.0

        data.append(audio)
        labels.append(label)

    # padding the audio's data
    data = pad_data(features=data, max_frames=max_samples)

    data = torch.cat(data, 0).to(dtype=torch.float32)
    data = data.unsqueeze(1)
    labels = torch.as_tensor(labels, dtype=torch.long)

    return data, labels
