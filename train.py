import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import argparse
from test import test
from src.dataset import create_dataloader
from src.utils import read_feature, feature_extraction_pipeline, read_features_files, choose_model
from src.data_augmentation import Mixup, Specmix, Cutmix
from src.models.utils import SaveBestModel
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from typing import Dict, Tuple, List, Union
from sklearn.metrics import classification_report

# making sure the experiments are reproducible
seed = 2109
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Adam,
    loss: torch.nn.CrossEntropyLoss,
    device: torch.device,
    mixer: Union[None, Mixup, Specmix, Cutmix]
) -> Tuple[float, float]:
    """
    Function responsible for the model training.

    Args:
        model (nn.Module): the created model.
        dataloader (DataLoader): the training dataloader.
        optimizer (torch.optim.Adam): the optimizer used.
        loss (torch.nn.CrossEntropyLoss): the loss function used.
        device (torch.device): which device to use.

    Returns:
        Tuple[float, float]: the training f1 and loss, respectively.
    """
    model.train()
    predictions = []
    targets = []
    train_loss = 0.0
    
    for index, (batch) in enumerate(dataloader, start=1):
        data = batch["features"].to(device)
        target = batch["labels"].to(device)
        optimizer.zero_grad()
        
        data = data.to(dtype=torch.float32)
        target = target.to(dtype=torch.float32)
        
        if not mixer is None:
            data, target = mixer(
                x=data,
                y=target
            )
            
        output = model(data)

        l = loss(output, target)
        train_loss += l.item()
        
        l.backward()
        optimizer.step()
        
        prediction = output.argmax(dim=-1, keepdim=True).to(dtype=torch.int)
        prediction = prediction.detach().cpu().numpy()
        predictions.extend(prediction.tolist())
        
        target = target.argmax(dim=-1, keepdim=True).to(dtype=torch.int)
        target = target.detach().cpu().numpy()
        targets.extend(target.tolist())
        
    train_loss = train_loss/index
    train_f1 = classification_report(
        targets,
        predictions,
        digits=6,
        output_dict=True,
        zero_division=0.0
    )
    train_f1 = train_f1["macro avg"]["f1-score"]
    return train_f1, train_loss

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss: torch.nn.CrossEntropyLoss,
    device: torch.device
) -> Tuple[float, float]:
    """
    Function responsible for the model evaluation.

    Args:
        model (nn.Module): the created model.
        dataloader (DataLoader): the validaiton dataloader.
        loss (torch.nn.CrossEntropyLoss): the loss function used.
        device (torch.device): which device to use.

    Returns:
        Tuple[float, float]: the validation f1 and loss, respectively.
    """
    model.eval()
    predictions = []
    targets = []
    validation_loss = 0.0
    validation_f1 = []
    
    with torch.inference_mode():
        for index, (batch) in enumerate(dataloader):
            data = batch["features"].to(device)
            target = batch["labels"].to(device)

            data = data.to(dtype=torch.float32)
            target = target.to(dtype=torch.float32)
                        
            output = model(data)
            
            l = loss(output, target)
            validation_loss += l.item()
            
            prediction = output.argmax(dim=-1, keepdim=True).to(dtype=torch.int)
            prediction = prediction.detach().cpu().numpy()
            predictions.extend(prediction.tolist())
            
            target = target.argmax(dim=-1, keepdim=True).to(dtype=torch.int)
            target = target.detach().cpu().numpy()
            targets.extend(target.tolist())
    
    validation_loss = validation_loss/index
    validation_f1 = classification_report(
        targets,
        predictions,
        digits=6,
        output_dict=True,
        zero_division=0.0
    )
    validation_f1 = validation_f1["macro avg"]["f1-score"]
    return validation_f1, validation_loss

def training_pipeline(
    training_data: List,
    validation_data: List,
    feature_config: Dict,
    wavelet_config: Dict,
    data_augmentation_config: Dict,
    model_config: Dict,
    mode: str,
    dataset: str
) -> None:
    total_folds = len(training_data)
    best_valid_f1, best_train_f1, best_test_f1 = [], [], []
        
    if dataset == "propor2022":
        if data_augmentation_config["target"] == "majority":
            data_augment_target = [0]
        elif data_augmentation_config["target"] == "minority":
            data_augment_target = [1, 2]
        elif data_augmentation_config["target"] == "all":
            data_augment_target = [0, 1, 2]
        else:
            raise ValueError("Invalid arguments for target. Should be 'all', 'majority' or 'minority")
    else:
        raise NotImplementedError
    
    # creating log folder
    log_path = os.path.join(os.getcwd(), "logs", dataset, mode)
    os.makedirs(log_path, exist_ok=True)
    logs = pd.DataFrame()
    
    feat_path = os.path.join(params["output_path"], params["dataset"])
        
    # reading training audio features
    X_test = read_feature(
        path=feat_path,
        name="X_test.pth",
    )
    
    y_test = read_feature(
        path=feat_path,
        name="y_test.pth",
    )
        
    for fold, (training, validation) in enumerate(zip(training_data, validation_data)):
        X_train, y_train = training
        X_valid, y_valid = validation
        
        # creating and defining the model
        device = torch.device("cuda" if torch.cuda.is_available and model_config["use_gpu"] else "cpu")
        
        model = choose_model(
            mode=mode,
            model_name=model_config["name"],
            dataset=dataset,
            device=device
        )
        
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=model_config["learning_rate"],
            weight_decay=0,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        loss = torch.nn.CrossEntropyLoss()
        scheduler = None
        mixer = None
        
        if model_config["use_lr_scheduler"]:
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        
        if "mixup" in data_augmentation_config["techniques"].keys():
            mixer = Mixup(
                alpha=data_augmentation_config["techniques"]["mixup"]["alpha"]
            )
        
        if "specmix" in data_augmentation_config["techniques"].keys():
            mixer = Specmix(
                p=data_augmentation_config["p"],
                min_band_size=data_augmentation_config["techniques"]["specmix"]["min_band_size"],
                max_band_size=data_augmentation_config["techniques"]["specmix"]["max_band_size"],
                max_frequency_bands=data_augmentation_config["techniques"]["specmix"]["max_frequency_bands"],
                max_time_bands=data_augmentation_config["techniques"]["specmix"]["max_time_bands"],
                device=device
            )
        
        if "cutmix" in data_augmentation_config["techniques"].keys():
            mixer = Cutmix(
                alpha=data_augmentation_config["techniques"]["cutmix"]["alpha"],
                p=data_augmentation_config["p"]
            )
                
        # creating the model checkpoint object
        sbm = SaveBestModel(
            output_dir=os.path.join(model_config["output_path"], dataset, mode, model_config["name"]),
            model_name=model_config["name"]
        )
        
        # creating the training dataloader
        training_dataloader = create_dataloader(
            X=X_train,
            y=y_train,
            feature_config=feature_config,
            wavelet_config=wavelet_config,
            data_augmentation_config=data_augmentation_config,
            num_workers=0,
            mode=mode,
            shuffle=False,
            training=True,
            batch_size=model_config["batch_size"],
            data_augment_target=data_augment_target,
            worker_init_fn=seed_worker,
            generator=g
        )
        
        # creating the validation dataloader
        validation_dataloader = create_dataloader(
            X=X_valid,
            y=y_valid,
            feature_config=feature_config,
            wavelet_config=wavelet_config,
            data_augmentation_config=data_augmentation_config,
            num_workers=0,
            mode=mode,
            shuffle=False,
            training=False,
            batch_size=model_config["batch_size"],
            data_augment_target=data_augment_target,
            worker_init_fn=seed_worker,
            generator=g
        )
        
        # creating the test dataloader
        test_dataloader = create_dataloader(
            X=X_test,
            y=y_test,
            feature_config=feat_config,
            wavelet_config=wavelet_config,
            data_augmentation_config=None,
            num_workers=0,
            mode=params["mode"],
            shuffle=False,
            training=False,
            batch_size=params["model"]["batch_size"],
            data_augment_target=None,
            worker_init_fn=seed_worker,
            generator=g
        )
        
        if total_folds != 1:
            print(); print("#" * 20)
            print(f"TRAINING FOLD: {fold}")
            print("#" * 20); print()
        else:
            print(); print("#" * 20)
            print(f"TRAINING")
            print("#" * 20); print()
            
        # training loop
        for epoch in range(1, model_config["epochs"] + 1):
            print(f"Epoch: {epoch}/{model_config['epochs']}")
            
            train_f1, train_loss = train(
                device=device,
                dataloader=training_dataloader,
                optimizer=optimizer,
                model=model,
                loss=loss,
                mixer=mixer
            )
            
            valid_f1, valid_loss = evaluate(
                device=device,
                dataloader=validation_dataloader,
                model=model,
                loss=loss
            )

            report = test(
                model=model,
                dataloader=test_dataloader,
                device=device
            )
            
            test_f1 = report["macro avg"]["f1-score"]
                        
            # saving the best model
            sbm(
                current_valid_f1=valid_f1,
                current_valid_loss=valid_loss,
                current_test_f1=test_f1,
                current_train_f1=train_f1,
                epoch=epoch,
                fold=fold,
                model=model,
                optimizer=optimizer
            )
            
            # updating learning rate
            if not scheduler is None:
                scheduler.step()
            
            row = pd.DataFrame({
                "epoch": [epoch],
                "train_f1": [train_f1],
                "train_loss": [train_loss],
                "validation_f1": [valid_f1],
                "validation_loss": [valid_loss]
            })
            
            logs = pd.concat([
                logs, row
            ], axis=0)
        
        # printing the best result
        print(); print("*" * 40);
        print(f"Epoch: {sbm.best_epoch}")
        print(f"Best F1-Score: {sbm.best_valid_f1}")
        print(f"Best Loss: {sbm.best_valid_loss}")
        print("*" * 40); print();
        
        best_train_f1.append(sbm.best_train_f1)
        best_valid_f1.append(sbm.best_valid_f1)
        best_test_f1.append(sbm.best_test_f1)
                
        logs = logs.reset_index(drop=True)
        logs.to_csv(
            path_or_buf=os.path.join(log_path, f"fold{fold if total_folds != 1 else ''}.csv"),
            sep=",",
            index=False
        )
        logs = pd.DataFrame()
    
    # printing the best result
    print(); print("#" * 40);
    print(f"Best Train F1-Score: {best_train_f1}")
    print(f"Best Validation F1-Score: {best_valid_f1}")
    print(f"Best Test F1-Score: {best_test_f1}")
    print("#" * 40); print();

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="the json configuration file path.")
    args = parser.parse_args()
    
    assert os.path.exists(args.config), "Configuration file does not exist!"
    
    # reading the parameters configuration file
    params = json.load(open(args.config, "r"))
    
    # parameters defination
    k_fold = None
    max_seconds = 16
    
    if "kfold" in params.keys():
        k_fold = params["kfold"]["num_k"]
    
    max_samples = max_seconds * int(params["sample_rate"])
    
    feat_config = params["feature"]
    feat_config["sample_rate"] = int(params["sample_rate"])
    data_augmentation_config = params["data_augmentation"]
    wavelet_config = params["wavelet"]
    
    feat_path = os.path.join(params["output_path"], params["dataset"])
    
    # feature extraction pipeline
    if params["overwrite"] or not os.path.exists(params["output_path"]):
        print(); print("EXTRACTING THE FEATURES..."); print();
                
        feature_extraction_pipeline(
            sample_rate=int(params["sample_rate"]),
            to_mono=params["to_mono"],
            dataset=params["dataset"],
            max_samples=max_samples,
            k_fold=k_fold,
            output_path=params["output_path"],
            input_path=params["input_path"]
        )
    
    # reading the previously extracted features
    training_data, validation_data = read_features_files(
        k_fold=k_fold,
        feat_path=feat_path
    )
    
    model_config = params["model"]
    
    print(); print("TRAINING THE MODEL...");
    
    # training step
    training_pipeline(
        training_data=training_data,
        validation_data=validation_data,
        feature_config=feat_config,
        wavelet_config=wavelet_config,
        data_augmentation_config=data_augmentation_config,
        model_config=model_config,
        mode=params["mode"],
        dataset=params["dataset"]
    )