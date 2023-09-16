import pandas as pd
import re
import os
import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.nn as nn
from torch.nn.functional import one_hot
from src.processing import split_data, processing
from src.models.aghajani import CNN
from src.models.qiuqiangkong import Transfer_CNN10, Transfer_CNN6
from typing import Optional, Union, Tuple, List


def pad_features(features: List, max_height: int, max_width: int) -> torch.Tensor:
    """
    Auxiliary function to pad the features.

    Args:
        features (List): the features that will be padded (mfcc, spectogram or mel_spectogram).
        max_height (int): the height max value.
        max_width (int): the width max value.

    Returns:
        List: the padded features.
    """
    features = [
        F.pad(f, (0, max_width - f.size(2), 0, max_height - f.size(1)))
        for f in features
    ]
    return features


def choose_model(
    mode: str, model_name: str, dataset: str, device: torch.device
) -> nn.Module:
    """
    Creates the model based on the given model_name and mode.

    Args:
        mode (str): which mode is running.
        model_name (str): the model name.
        dataset (str): the dataset name.
        device (torch.device): the device where the model will be ran.

    Raises:
        ValueError: if the chosen model is not supported in the given mode.

    Returns:
        nn.Module: the created model.
    """
    if dataset == "coraa":
        num_classes = 3
    elif dataset == "emodb":
        num_classes = 7
    elif dataset == "ravdess":
        num_classes = 8
    elif dataset == "savee":
        num_classes = 7
    else:
        raise ValueError("Invalid dataset")

    if mode == "mode_1":
        if model_name == "aghajani":
            model = CNN(input_channels=1, num_classes=num_classes).to(device)
        elif model_name == "qiuqiangkong":
            model = Transfer_CNN10(
                input_channels=1,
                num_classes=num_classes,
                load_pretrained=False,
                freeze_base=False,
            ).to(device)
    elif mode == "mode_2":
        if model_name == "aghajani":
            model = CNN(input_channels=4, num_classes=num_classes).to(device)
        elif model_name == "qiuqiangkong":
            model = Transfer_CNN6(
                input_channels=4,
                num_classes=num_classes,
                load_pretrained=False,
                freeze_base=False,
            ).to(device)
    else:
        raise ValueError("Unknown mode")

    return model


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


def convert_frequency_to_mel(f: float) -> float:
    """
    Extracted from: https://github.com/iver56/audiomentations/blob/24748c64d85499aefce5e21ce91d59cf1d658374/audiomentations/core/utils.py

    Convert f hertz to mels
    https://en.wikipedia.org/wiki/Mel_scale#Formula
    """
    return 2595.0 * math.log10(1.0 + f / 700.0)


def convert_mel_to_frequency(m: Union[float, np.array]) -> Union[float, np.array]:
    """
    Extracted from: https://github.com/iver56/audiomentations/blob/24748c64d85499aefce5e21ce91d59cf1d658374/audiomentations/core/utils.py

    Convert m mels to hertz
    https://en.wikipedia.org/wiki/Mel_scale#History_and_other_formulas
    """
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def labels_mapping(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    Converts the labels string to int.

    Args:
        df (pd.DataFrame): the dataset's dataframe.
        dataset (str): which dataset is being used.

    Returns:
        pd.DataFrame: the dataframe with labels mapped.
    """
    if dataset == "coraa":
        df["label"] = df["label"].replace(
            {"neutral": 0, "non-neutral-male": 1, "non-neutral-female": 2}
        )
    elif dataset == "emodb":
        df["label"] = df["label"].replace(
            {
                "neutral": 0,
                "anger": 1,
                "boredom": 2,
                "disgust": 3,
                "anxiety/fear": 4,
                "happiness": 5,
                "sadness": 6,
            }
        )
    elif dataset == "ravdess":
        df["label"] = df["label"].replace(
            {
                "neutral": 0,
                "angry": 1,
                "calm": 2,
                "disgust": 3,
                "fearful": 4,
                "happy": 5,
                "sad": 6,
                "surprised": 7,
            }
        )
    elif dataset == "savee":
        df["label"] = df["label"].replace(
            {
                "neutral": 0,
                "anger": 1,
                "disgust": 2,
                "fear": 3,
                "happiness": 4,
                "sadness": 5,
                "surprise": 6,
            }
        )

    return df


def read_feature(path: str, fold: Union[int, None], name: str) -> torch.Tensor:
    """
    Reads the saved feature.

    Args:
        path (str): the feature's folder path.
        fold (int | None): which folder is it (None if kfold is not used).
        name (str): the name of the feature.

    Returns:
        torch.Tensor: the read feature.
    """
    if not fold is None:
        feature = torch.load(os.path.join(path, f"fold{fold}", name))
    else:
        feature = torch.load(os.path.join(path, name))

    return feature


def prepare_coraa_test_dataframe(
    path: str = "/media/greca/HD/Datasets/PROPOR 2022/",
) -> pd.DataFrame:
    """
    Prepares the CORAA's testing dataset to have the same structure as
    the training dataset.

    Args:
        path (str, optional): the test dataset path. Defaults to "/media/greca/HD/Datasets/PROPOR 2022/".

    Returns:
        pd.DataFrame: the formatted test dataset.
    """
    df = pd.read_csv(os.path.join(path, "test_ser_metadata.csv"), sep=",")
    df.columns = ["wav_file", "label", "file"]
    df["file"] = df["file"].apply(lambda x: os.path.join(path, "test_ser", x))
    return df.reset_index(drop=True)


def create_ravdess_train_dataframe(
    path: str = "/media/greca/HD/Datasets/RAVDESS/",
) -> pd.DataFrame:
    """
    Creates a RAVDESS's pandas DataFrame containing all the training files.

    Args:
        path (str): the path to the CSV file.

    Returns:
        df (pd.DataFrame): the pandas DataFrame.
    """
    df = pd.DataFrame()
    actors = [f"Actor_0{i}" if i < 10 else f"Actor_{i}" for i in range(1, 25)]

    for actor in actors:
        wav_files = [
            file
            for file in os.listdir(os.path.join(path, actor))
            if file.endswith(".wav")
        ]

        for wav in wav_files:
            wav_file = os.path.basename(wav)
            label = wav_file.split("-")[2]

            if label == "01":
                label = "neutral"
            elif label == "02":
                label = "calm"
            elif label == "03":
                label = "happy"
            elif label == "04":
                label = "sad"
            elif label == "05":
                label = "angry"
            elif label == "06":
                label = "fearful"
            elif label == "07":
                label = "disgust"
            elif label == "08":
                label = "surprised"

            row = pd.DataFrame(
                {
                    "file": [os.path.join(path, actor, wav)],
                    "label": [label],
                    "wav_file": [wav],
                }
            )

            df = pd.concat([df, row], axis=0)

    return df.reset_index(drop=True)


def create_emodb_train_dataframe(
    path: str = "/media/greca/HD/Datasets/EmoDB/",
) -> pd.DataFrame:
    """
    Creates a EmoDB's pandas DataFrame containing all the training files.

    Args:
        path (str): the path to the CSV file.

    Returns:
        df (pd.DataFrame): the pandas DataFrame.
    """
    wav_files = [
        file for file in os.listdir(os.path.join(path, "wav")) if file.endswith(".wav")
    ]
    df = pd.DataFrame()

    for wav in wav_files:
        wav_file = os.path.basename(wav)
        speaker_file = wav_file[:2]
        text_code = wav_file[2:5]
        label = wav_file[5].lower()
        version = wav_file[6]

        if label == "w":
            label = "anger"
        elif label == "l":
            label = "boredom"
        elif label == "e":
            label = "disgust"
        elif label == "a":
            label = "anxiety/fear"
        elif label == "f":
            label = "happiness"
        elif label == "t":
            label = "sadness"
        elif label == "n":
            label = "neutral"

        row = pd.DataFrame(
            {
                "file": [os.path.join(path, "wav", wav)],
                "label": [label],
                "wav_file": [wav],
            }
        )

        df = pd.concat([df, row], axis=0)

    return df.reset_index(drop=True)


def create_savee_train_dataframe(
    path: str = "/media/greca/HD/Datasets/SAVEE/",
) -> pd.DataFrame:
    """
    Creates a SAVEE's pandas DataFrame containing all the training files.

    Args:
        path (str): the path to the CSV file.

    Returns:
        df (pd.DataFrame): the pandas DataFrame.
    """
    wav_files = [file for file in os.listdir(path) if file.endswith(".wav")]
    df = pd.DataFrame()

    for wav in wav_files:
        wav_file = os.path.basename(wav)
        speaker_file = wav_file[:2]
        label = wav_file[3:]
        label = re.findall("[a-zA-Z]+", str(label))[0]

        if label == "a":
            label = "anger"
        elif label == "d":
            label = "disgust"
        elif label == "f":
            label = "fear"
        elif label == "h":
            label = "happiness"
        elif label == "sa":
            label = "sadness"
        elif label == "n":
            label = "neutral"
        elif label == "su":
            label = "surprise"

        row = pd.DataFrame(
            {"file": [os.path.join(path, wav)], "label": [label], "wav_file": [wav]}
        )

        df = pd.concat([df, row], axis=0)

    return df.reset_index(drop=True)


def create_coraa_train_dataframe(
    path: str = "/media/greca/HD/Datasets/PROPOR 2022/",
) -> pd.DataFrame:
    """
    Creates a CORAA's pandas DataFrame containing
    all the training files using the same structure as the
    `test_ser_metadata.csv` file.

    Args:
        path (str): the path to the CSV file.

    Returns:
        df (pd.DataFrame): the pandas DataFrame.
    """
    wav_files = [
        file
        for file in os.listdir(os.path.join(path, "data_train/train"))
        if file.endswith(".wav")
    ]
    df = pd.DataFrame()

    for wav in wav_files:
        wav_file = os.path.basename(wav)
        wav_file = wav_file.split("/")[0]
        label = wav_file.split("_")[-1].replace(".wav", "")

        row = pd.DataFrame(
            {
                "file": [os.path.join(path, "data_train/train", wav_file)],
                "label": [label],
                "wav_file": [wav_file],
            }
        )

        df = pd.concat([df, row], axis=0)

    return df.reset_index(drop=True)


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


def read_feature(path: str, name: str, fold: Optional[int] = None) -> torch.Tensor:
    """
    Reads the saved feature.

    Args:
        path (str): the feature's folder path.
        fold (int | None): which folder is it (None if kfold is not used).
        name (str): the name of the feature.

    Returns:
        torch.Tensor: the read feature.
    """
    if not fold is None:
        feature = torch.load(os.path.join(path, f"fold{fold}", name))
    else:
        feature = torch.load(os.path.join(path, name))

    return feature


def feature_extraction_pipeline(
    dataset: str,
    to_mono: bool,
    sample_rate: int,
    max_samples: int,
    k_fold: int,
    output_path: str,
    input_path: str,
    apply_one_hot_encoder: bool = True,
) -> None:
    """
    Feature extraction pipeline.

    Args:
        dataset (str): which dataset is being used.
        to_mono (bool): convert the audio to mono or not.
        sample_rate (int): audio's sample rate.
        max_samples (int): the maximum number of data samples.
        k_fold (int): how many folds the data will be splitted into.
        output_path (str): the output path where the features will be saved.
        input_path (str): the dataset input path.
        apply_one_hot_encoder (bool, optional): if one hot encoder will be
                                                applied or not. Defaults to True.
    """
    # reading the training dataset
    if dataset == "coraa":
        train_df = create_coraa_train_dataframe(path=input_path)
    elif dataset == "emodb":
        train_df = create_emodb_train_dataframe(path=input_path)
    elif dataset == "ravdess":
        train_df = create_ravdess_train_dataframe(path=input_path)
    elif dataset == "savee":
        train_df = create_savee_train_dataframe(path=input_path)

    train_df = labels_mapping(df=train_df, dataset=dataset)

    # preprocessing the training dataset
    X_train, y_train = processing(
        df=train_df, to_mono=to_mono, sample_rate=sample_rate, max_samples=max_samples
    )

    # splitting the training dataset into training and validation (and saving)
    split_data(
        X=X_train,
        y=y_train,
        dataset=dataset,
        output_path=output_path,
        k_fold=k_fold,
        apply_one_hot_encoder=apply_one_hot_encoder,
    )

    # reading the test dataset (only for CORAA)
    if dataset == "coraa":
        test_df = prepare_coraa_test_dataframe(path=input_path)
        test_df = labels_mapping(df=test_df, dataset=dataset)

        # preprocessing the test dataset
        X_test, y_test = processing(
            df=test_df,
            to_mono=to_mono,
            sample_rate=sample_rate,
            max_samples=max_samples,
        )

        if apply_one_hot_encoder:
            num_classes = 3
            y_test = one_hot_encoder(labels=y_test, num_classes=num_classes)

        # saving the test features
        folder_path = os.path.join(output_path, dataset)

        save(path=folder_path, name="X_test", tensor=X_test)
        save(path=folder_path, name="y_test", tensor=y_test)


def read_features_files(k_fold: Union[int, None], feat_path: str) -> Tuple[List, List]:
    """
    Read the features files.

    Args:
        k_fold (Union[int, None]): which fold is being read.
        feat_path (str): the extracted features path.

    Returns:
        Tuple[List, List]: the training and validation features, respectively.
    """
    training_data = []
    validation_data = []

    # reading all the previously extracted features
    if not k_fold is None:
        for fold in range(k_fold):
            # reading training audio features
            X_train = read_feature(
                path=feat_path,
                fold=fold,
                name="X_train.pth",
            )

            y_train = read_feature(
                path=feat_path,
                fold=fold,
                name="y_train.pth",
            )

            training_data.append((X_train, y_train))

            # reading the validation audio features
            X_valid = read_feature(
                path=feat_path,
                fold=fold,
                name="X_valid.pth",
            )

            y_valid = read_feature(
                path=feat_path,
                fold=fold,
                name="y_valid.pth",
            )

            validation_data.append((X_valid, y_valid))
    else:
        # reading training audio features
        X_train = read_feature(
            path=feat_path,
            name="X_train.pth",
        )

        y_train = read_feature(
            path=feat_path,
            name="y_train.pth",
        )

        training_data.append((X_train, y_train))

        # reading the validation audio features
        X_valid = read_feature(
            path=feat_path,
            name="X_valid.pth",
        )

        y_valid = read_feature(
            path=feat_path,
            name="y_valid.pth",
        )

        validation_data.append((X_valid, y_valid))

    return training_data, validation_data
