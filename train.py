import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from src.dataset import create_dataloader
from src.utils import feature_extraction_pipeline, read_features_files
from src.models.cnn import CNN
from src.models.utils import SaveBestModel
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from typing import Dict, Tuple, List

# making sure the experiments are reproducible
seed = 2109
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Adam,
    loss: torch.nn.CrossEntropyLoss,
    device: torch.device
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
    train_loss = 0.0
    train_f1 = 0.0
    
    for batch in dataloader:
        data = batch["features"].to(device)
        target = batch["labels"].to(device)
        optimizer.zero_grad()
        
        data = data.to(dtype=torch.float32)
        target = target.to(dtype=torch.float32)
        
        output = model(data)

        l = loss(output, target)
            
        train_loss += l.item()
        
        l.backward()
        optimizer.step()
        
        prediction = output.argmax(dim=-1, keepdim=False).to(dtype=torch.int)
        target = target.argmax(dim=-1, keepdim=False).to(dtype=torch.int)
        
        train_f1 += f1_score(
            target.detach().cpu().numpy(),
            prediction.detach().cpu().numpy(),
            average="macro"
        )
    
    train_loss /= len(dataloader)
    train_f1 /= len(dataloader)
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
    validation_loss = 0.0
    validation_f1 = 0.0
    
    with torch.inference_mode():
        for batch in dataloader:
            data = batch["features"].to(device)
            target = batch["labels"].to(device)

            data = data.to(dtype=torch.float32)
            target = target.to(dtype=torch.float32)
                        
            output = model(data)
            
            l = loss(output, target)
            validation_loss += l.item()
            
            prediction = output.argmax(dim=-1, keepdim=False).to(dtype=torch.int)
            target = target.argmax(dim=-1, keepdim=False).to(dtype=torch.int)
            
            validation_f1 += f1_score(
                target.detach().cpu().numpy(),
                prediction.detach().cpu().numpy(),
                average="macro"
            )
    
    validation_loss /= len(dataloader)
    validation_f1 /= len(dataloader)
    return validation_f1, validation_loss

def training_pipeline(
    training_data: List,
    validation_data: List,
    feature_config: Dict,
    wavelet_config: Dict,
    model_name: str,
    output_dir: str,
    epochs: int
):
    total_folds = len(training_data)
    
    for fold, (training, validation) in enumerate(zip(training_data, validation_data)):
        X_train, y_train = training
        X_valid, y_valid = validation
        
        # creating and defining the model
        model = CNN()
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=0.001
        )
        loss = torch.nn.CrossEntropyLoss()
        device = torch.device("cuda" if torch.cuda.is_available else "cpu")
        
        # creating the model checkpoint object
        sbm = SaveBestModel(
            output_dir=output_dir,
            model_name=model_name
        )
        
        # creating the training dataloader
        training_dataloader = create_dataloader(
            X=X_train,
            y=y_train,
            feature_config=feature_config,
            wavelet_config=wavelet_config,
            num_workers=0,
            shuffle=True,
            training=True,
            batch_size=32
        )
        
        # creating the validation dataloader
        validation_dataloader = create_dataloader(
            X=X_valid,
            y=y_valid,
            feature_config=feature_config,
            wavelet_config=wavelet_config,
            num_workers=0,
            shuffle=True,
            training=False,
            batch_size=32
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
        for epoch in range(1, epochs+1):
            print(f"Epoch: {epoch}/{epochs}")
            
            train_f1, train_loss = train(
                device=device,
                dataloader=training_dataloader,
                optimizer=optimizer,
                model=model,
                loss=loss
            )
            
            valid_f1, valid_loss = evaluate(
                device=device,
                dataloader=validation_dataloader,
                model=model,
                loss=loss
            )

            # saving the best model
            sbm(
                current_valid_f1=valid_f1,
                current_valid_loss=valid_loss,
                epoch=epoch,
                fold=fold,
                model=model,
                optimizer=optimizer
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
    
    # reading the previously extracted features
    training_data, validation_data = read_features_files(
        k_fold=k_fold,
        feat_path=feat_path
    )
    
    model_output_dir = os.path.join("./checkpoints", "cnn")
    
    # training step
    training_pipeline(
        training_data=training_data,
        validation_data=validation_data,
        feature_config=feat_config,
        wavelet_config=wavelet_config,
        epochs=50,
        model_name="cnn",
        output_dir=model_output_dir
    )