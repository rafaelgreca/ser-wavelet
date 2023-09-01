import argparse
import os
import json
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from src.utils import read_feature, choose_model
from src.dataset import create_dataloader
from typing import Dict

def test(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict:
    """
    Function responsible for the model testing in the test dataset.

    Args:
        model (nn.Module): the created model.
        dataloader (DataLoader): the test dataloader.
        device (torch.device): which device to use.

    Returns:
        Dict: the test metrics score.
    """
    model.eval()
    predictions = []
    targets = []
    
    with torch.inference_mode():
        for batch in dataloader:
            data = batch["features"].to(device)
            target = batch["labels"].to(device)

            data = data.to(dtype=torch.float32)
            target = target.to(dtype=torch.float32)
            
            output = model(data)
            
            # prediction = output.argmax(dim=-1, keepdim=True).to(dtype=torch.int)
            prediction = output.detach().cpu().numpy()
            predictions.extend(prediction.tolist())
            
            #  target = target.argmax(dim=-1, keepdim=True).to(dtype=torch.int)
            target = target.detach().cpu().numpy()
            targets.extend(target.tolist())
    
    roc_auc = roc_auc_score(
        y_true=targets,
        y_score=predictions,
        average="macro",
        multi_class="ovr"
    )
    
    predictions = torch.Tensor(predictions).argmax(dim=-1, keepdim=True).to(dtype=torch.int)
    predictions = predictions.numpy().tolist()
    
    targets = torch.Tensor(targets).argmax(dim=-1, keepdim=True).to(dtype=torch.int)
    targets = targets.numpy().tolist()
    
    f1_score = classification_report(
        y_true=targets,
        y_pred=predictions,
        digits=4,
        output_dict=True,
        zero_division=0.0
    )["macro avg"]["f1-score"]
    
    accuracy = accuracy_score(
        y_true=targets, 
        y_pred=predictions
    )
        
    metrics = {
        "f1-score macro": f1_score,
        "unweighted accuracy": accuracy,
        "roc-auc macro": roc_auc
    }
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="the json configuration file path.")
    args = parser.parse_args()
    
    assert os.path.exists(args.config), "Configuration file does not exist!"
    
    # reading the parameters configuration file
    params = json.load(open(args.config, "r"))
    
    # parameters defination
    k_fold = None
    
    if params["dataset"].lower() == "propor2022":
        max_seconds = 16
    elif params["dataset"].lower() == "emodb":
        max_seconds = 10
    elif params["dataset"].lower() == "ravdess":
        max_seconds = 6
    elif params["dataset"].lower() == "savee":
        max_seconds = 8
        
    if "kfold" in params.keys():
        k_fold = params["kfold"]["num_k"]
    
    feat_config = params["feature"]
    feat_config["sample_rate"] = int(params["sample_rate"])
    wavelet_config = params["wavelet"]
    
    if os.path.exists(params["output_path"]):
        feat_path = os.path.join(params["output_path"], params["dataset"])
        
        # reading training audio features
        if params["dataset"].lower() == "propor2022":
            X_test = read_feature(
                path=feat_path,
                name="X_test.pth",
            )
            
            y_test = read_feature(
                path=feat_path,
                name="y_test.pth",
            )
                    
        device = torch.device("cuda") if params["model"]["use_gpu"] else torch.device("cpu")
          
        output_path = params["model"]["output_path"]
        model_name = params["model"]["name"]
        
        model = choose_model(
            mode=params["mode"],
            model_name=model_name,
            device=device,
            dataset=params["dataset"]
        )
                        
        if not k_fold is None:
            for fold in range(k_fold):
                if params["dataset"].lower() != "propor2022":
                    X_test = read_feature(
                        path=os.path.join(feat_path, f"fold{fold}"),
                        name="X_valid.pth",
                    )
                    
                    y_test = read_feature(
                        path=os.path.join(feat_path, f"fold{fold}"),
                        name="y_valid.pth",
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
                    data_augment_target=None
                )
                
                # loading the trained model parameters
                model.load_state_dict(
                    torch.load(
                        os.path.join(output_path, params["dataset"], params["mode"], model_name, f"{model_name}_fold{fold}.pth")
                    )["model_state_dict"]
                )
                
                scores = test(
                    model=model,
                    dataloader=test_dataloader,
                    device=device
                )
                
                print(); print("#" * 20)
                print(f"FOLD: {fold}")
                print(f"F1-Score macro: {scores['f1-score macro']}")
                print(f"Unweighted accuracy: {scores['unweighted accuracy']}")
                print(f"ROC-AUC macro: {scores['roc-auc macro']}")
                print("#" * 20); print()
                
        else:
            if params["dataset"].lower() != "propor2022":
                X_test = read_feature(
                    path=feat_path,
                    name="X_valid.pth",
                )
                
                y_test = read_feature(
                    path=feat_path,
                    name="y_valid.pth",
                )
                
            # loading the trained model parameters
            model.load_state_dict(
                torch.load(
                    os.path.join(output_path, params["feature"]["dataset"], params["mode"], model_name, f"{model_name}.pth")
                )["model_state_dict"]
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
                data_augment_target=None
            )
            
            scores = test(
                model=model,
                dataloader=test_dataloader,
                device=device
            )
            
            print(); print("#" * 20)
            print(f"REPORT:")
            print(f"F1-Score macro: {scores['f1-score macro']}")
            print(f"Unweighted accuracy: {scores['unweighted accuracy']}")
            print(f"ROC-AUC macro: {scores['roc-auc macro']}")
            print("#" * 20); print()      
    else:
        raise "Please run the feature extraction algorithm first."