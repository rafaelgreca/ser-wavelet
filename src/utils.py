import pandas as pd
import os
import torch
from src.processing import split_data, processing
from typing import Optional, Union, Tuple, List

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
    df.columns = ["wav_file", "label", "file"]
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

def save(
    path: str,
    name: str,
    tensor: torch.Tensor
) -> None:
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
    
def read_feature(
    path: str,
    name: str,
    fold: Optional[int] = None
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

def feature_extraction_pipeline(
    dataset: str,
    to_mono: bool,
    sample_rate: int,
    max_samples: int,
    k_fold: int,
    output_path: str,
    input_path: str
) -> None:
    # reading the training dataset
    train_df = create_propor_train_dataframe(
        path=input_path
    )
    train_df = labels_mapping(
        df=train_df,
        dataset=dataset
    )
    
    # preprocessing the training dataset
    X_train, y_train = processing(
        df=train_df,
        to_mono=to_mono,
        sample_rate=sample_rate,
        max_samples=max_samples
    )
    
    # splitting the training dataset into training and validation (and saving)
    split_data(
        X=X_train,
        y=y_train,
        dataset=dataset,
        output_path=output_path,
        k_fold=k_fold
    )
    
    # reading the test dataset
    test_df = prepare_propor_test_dataframe(
        path=input_path
    )
    test_df = labels_mapping(
        df=test_df,
        dataset=dataset
    )
        
    # preprocessing the test dataset
    X_test, y_test = processing(
        df=test_df,
        to_mono=to_mono,
        sample_rate=sample_rate,
        max_samples=max_samples
    )
    
    # saving the test features
    folder_path = os.path.join(output_path, dataset)
        
    save(path=folder_path, name="X_test", tensor=X_test)
    save(path=folder_path, name="y_test", tensor=y_test)

def read_features_files(
    k_fold: Union[int, None],
    feat_path: str
) -> Tuple[List, List]:
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