import argparse
import os
import json
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from src.utils import read_feature, choose_model
from src.dataset import create_dataloader

def test(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> str:
    """
    Function responsible for the model testing in the test dataset.

    Args:
        model (nn.Module): the created model.
        dataloader (DataLoader): the test dataloader.
        device (torch.device): which device to use.

    Returns:
        str: the test classification report.
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
            
            prediction = output.argmax(dim=-1, keepdim=True).to(dtype=torch.int)
            prediction = prediction.detach().cpu().numpy()
            predictions.extend(prediction.tolist())
            
            target = target.argmax(dim=-1, keepdim=True).to(dtype=torch.int)
            target = target.detach().cpu().numpy()
            targets.extend(target.tolist())
    
    class_report = classification_report(
        targets,
        predictions,
        digits=4,
        output_dict=True
    )
    return class_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="the json configuration file path.")
    args = parser.parse_args()
    
    assert os.path.exists(args.config), "Configuration file does not exist!"
    
    # reading the parameters configuration file
    params = json.load(open(args.config, "r"))
    
    # parameters defination
    k_fold = None
    max_seconds = 15
    
    if "kfold" in params.keys():
        k_fold = params["kfold"]["num_k"]
    
    feat_config = params["feature"]
    feat_config["sample_rate"] = int(params["sample_rate"])
    wavelet_config = params["wavelet"]
    
    if os.path.exists(params["output_path"]):
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
        
        device = torch.device("cuda") if params["model"]["use_gpu"] else torch.device("cpu")
          
        output_path = params["model"]["output_path"]
        model_name = params["model"]["name"]
        
        model = choose_model(
            mode=params["mode"],
            model_name=model_name,
            device=device
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
        
        if not k_fold is None:
            for fold in range(k_fold):
                # loading the trained model parameters
                model.load_state_dict(
                    torch.load(
                        os.path.join(output_path, params["dataset"], params["mode"], model_name, f"{model_name}_fold{fold}.pth")
                    )["model_state_dict"]
                )
                
                fold_report = test(
                    model=model,
                    dataloader=test_dataloader,
                    device=device
                )
                
                print(); print("#" * 20)
                print(f"FOLD: {fold}")
                print(f"REPORT:")
                print(fold_report)
                print("#" * 20); print()
                
        else:
            # loading the trained model parameters
            model.load_state_dict(
                torch.load(
                    os.path.join(output_path, params["feature"]["dataset"], params["mode"], model_name, f"{model_name}.pth")
                )["model_state_dict"]
            )
            
            report = test(
                model=model,
                dataloader=test_dataloader,
                device=device
            )
            
            print(); print("#" * 20)
            print(f"REPORT:")
            print(report)
            print("#" * 20); print()      
    else:
        raise "Please run the feature extraction algorithm first."