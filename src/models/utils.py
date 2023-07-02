import os
import numpy as np
import torch
import torch.nn as nn
from typing import Union

# All credits to: https://discuss.pytorch.org/t/same-implementation-different-results-between-keras-and-pytorch-lstm/39146
def weight_init(
    m: torch.nn.Module
):
    """
    Initalize all the weights in the PyTorch model to be the same as Keras.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    if isinstance(m, nn.LSTM):
        nn.init.xavier_uniform_(m.weight_ih_l0)
        nn.init.orthogonal_(m.weight_hh_l0)
        nn.init.zeros_(m.bias_ih_l0)
        nn.init.zeros_(m.bias_hh_l0)
        
class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self,
        output_dir: str,
        model_name: str
    ) -> None:
        """
        Args:
            output_dir (str): the output folder directory.
            model_name (str): the model's name.
        """
        self.best_valid_loss = float(np.Inf)
        self.best_valid_f1 = float(np.NINF)
        self.best_test_f1 = float(np.NINF)
        self.best_train_f1 = float(np.NINF)
        self.output_dir = output_dir
        self.model_name = model_name
        self.save_model = False
        self.best_epoch = -1
        os.makedirs(self.output_dir, exist_ok=True)

    def __call__(
        self,
        current_valid_loss: float,
        current_valid_f1: float,
        current_test_f1: float,
        current_train_f1: float,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim,
        fold: Union[int, None]
    ) -> None:
        """
        Saves the best trained model.

        Args:
            current_valid_loss (float): the current validation loss value.
            current_valid_f1 (float): the current validation f1 score value.
            current_test_f1 (float): the current test f1 score value.
            current_train_f1 (float): the current train f1 score value.
            epoch (int): the current epoch.
            model (nn.Module): the trained model.
            optimizer (torch.optim): the optimizer objet.
            fold (Union[int, None]): the current fold.
        """
        if current_valid_f1 > self.best_valid_f1:
            self.best_valid_loss = current_valid_loss
            self.best_valid_f1 = current_valid_f1
            self.best_test_f1 = current_test_f1
            self.best_train_f1 = current_train_f1
            self.best_epoch = epoch
            self.save_model = True
        
        if self.save_model:
            print("\nSaving model...")
            print(f"Epoch: {epoch}")
            print(f"Train F1-Score: {current_train_f1:1.6f}")
            print(f"Validation F1-Score: {current_valid_f1:1.6f}")
            print(f"Validation Loss: {current_valid_loss:1.6f}")
            print(f"Test F1-Score: {current_test_f1:1.6f}\n")
            
            if not fold is None:
                path = os.path.join(self.output_dir, f"{self.model_name}_fold{fold}.pth")
            else:
                path = os.path.join(self.output_dir, f"{self.model_name}.pth")
                
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
                },
                path
            )
            self.save_model = False