import pandas as pd
import os
import torch
from torch.nn.functional import one_hot
from typing import Union

def labels_mapping(
    df: pd.DataFrame,
    dataset: str
) -> pd.DataFrame:
    """
    Converts the labels string to int.

    Args:
        df (pd.DataFrame): the dataset's dataframe.
        dataset (str): which dataset is being used.

    Returns:
        pd.DataFrame: the dataframe with labels mapped.
    """
    if dataset == "propor2022":
        df["label"] = df["label"].replace({
            "neutral": 0,
            "non-neutral-male": 1,
            "non-neutral-female": 2
        })
    elif dataset == "tess":
        df["label"] = df["label"].replace({
            "angry": 0,
            "disgust": 1,
            "fear": 2,
            "happy": 3,
            "neutral": 4,
            "surprised": 5,
            "sad": 6
        })
    
    return df

def one_hot_encoder(
    labels: torch.Tensor,
    num_classes: int = -1
) -> torch.Tensor:
    """
    Encode the labels into the one hot format.

    Args:
        labels (torch.Tensor): the data labels.
        num_classes (int): the number of classes present in the data.

    Returns:
        torch.Tensor: the data labels one hot encoded.
    """
    return one_hot(labels, num_classes=num_classes)

def read_feature(
    path: str,
    fold: Union[int, None],
    name: str
) -> torch.Tensor:
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
        feature = torch.load(
            os.path.join(path, f"fold{fold}", name)
        )
    else:
        feature = torch.load(
            os.path.join(path, name)
        )
    
    return feature

def prepare_propor_test_dataframe(
    path: str = "/media/greca/HD/Datasets/PROPOR 2022/"
) -> pd.DataFrame:
    """
    Prepares the PROPOR 2022's testing dataset to have the same structure as
    the training dataset.

    Args:
        path (str, optional): the test dataset path. Defaults to "/media/greca/HD/Datasets/PROPOR 2022/".

    Returns:
        pd.DataFrame: the formatted test dataset.
    """
    df = pd.read_csv(os.path.join(path, "test_ser_metadata.csv"), sep=",")
    df["file"] = df["file"].apply(lambda x: os.path.join(path, "test_ser", x))    
    return df.reset_index(drop=True)

def create_propor_train_dataframe(
    path: str = "/media/greca/HD/Datasets/PROPOR 2022/"
) -> pd.DataFrame:
    """
    Creates a PROPOR 2022's pandas DataFrame containing
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
        
        row = pd.DataFrame({
            "file": [os.path.join(path, "data_train/train", wav_file)],
            "label": [label],
            "wav_file": [wav_file]
        })
        
        df = pd.concat(
            [df, row],
            axis=0
        )
    
    return df.reset_index(drop=True)

def read_csv(
    path: str,
    separator: str
) -> pd.DataFrame:
    """
    Reads a CSV file and returns a pandas DataFrame.
    
    Args:
        path (str): the path to the CSV file.
        separator (str): the csv string separator.
    
    Returns:
        df (pd.DataFrame): the pandas DataFrame.
    """
    return pd.read_csv(path, sep=separator)