import json
import os
import random
import numpy as np
import torch
from src.dataset import create_dataloader
from src.utils import create_propor_train_dataframe, prepare_propor_test_dataframe, labels_mapping, save, read_feature
from src.processing import processing, split_data
from typing import Union, Dict

# Making sure the experiments are reproducible
seed = 2109
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def feature_extraction_pipeline(
    dataset: str,
    to_mono: bool,
    sample_rate: int,
    max_samples: int,
    k_fold: int,
    output_path: str,
    input_path: str
):
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

def training_pipeline(
    k_fold: Union[int, None],
    feat_path: str,
    feature_config: Dict,
    wavelet_config: Dict,
    batch_size: int
):
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
            
            # creating the training dataloader
            training_dataloader = create_dataloader(
                X=X_train,
                y=y_train,
                feature_config=feature_config,
                wavelet_config=wavelet_config,
                batch_size=batch_size,
                num_workers=0,
                shuffle=True,
                training=True
            )
            
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

if __name__ == "__main__":
    # reading the parameters configuration file
    params = json.load(open("./config.json", "r"))
    
    # parameters defination
    k_fold = None
    max_seconds = 15
    
    if "kfold" in params["feature"].keys():
        k_fold = params["feature"]["kfold"]["num_k"]
    
    max_samples = max_seconds * int(params["feature"]["sample_rate"])
    
    feat_config = params["feature"]["config"]
    feat_config["sample_rate"] = int(params["feature"]["sample_rate"])
    
    wavelet_config = params["feature"]["wavelet"]
    feat_path = os.path.join(params["feature"]["output_path"], params["feature"]["dataset"])
    
    # feature extraction pipeline
    if params["feature"]["overwrite"] or not os.path.exists(params["feature"]["output_path"]):
        print(); print("EXTRACTING THE FEATURES..."); print();
                
        feature_extraction_pipeline(
            sample_rate=int(params["feature"]["sample_rate"]),
            to_mono=params["feature"]["to_mono"],
            dataset=params["feature"]["dataset"],
            max_samples=max_samples,
            k_fold=k_fold,
            output_path=params["feature"]["output_path"],
            input_path=params["feature"]["input_path"]
        )
            
    