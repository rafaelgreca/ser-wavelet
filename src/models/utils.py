import os
import numpy as np
import torch
import torch.nn as nn
from typing import Union

def interpolate(x, ratio):
    """Interpolate data in time domain. This is used to compensate the 
    resolution reduction in downsampling of a CNN.
    
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate

    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output, frames_num):
    """Pad framewise_output to the same length as input frames. The pad value 
    is the same as the value of the last frame.

    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad

    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1 :, :].repeat(1, frames_num - framewise_output.shape[1], 1)
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output

def do_mixup(x, mixup_lambda):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes 
    (1, 3, 5, ...).

    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)

    Returns:
      out: (batch_size, ...)
    """
    out = (x[0 :: 2].transpose(0, -1) * mixup_lambda[0 :: 2] + \
        x[1 :: 2].transpose(0, -1) * mixup_lambda[1 :: 2]).transpose(0, -1)
    return out

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
        self.output_dir = output_dir
        self.model_name = model_name
        self.save_model = False
        self.best_epoch = -1
        os.makedirs(self.output_dir, exist_ok=True)

    def __call__(
        self,
        current_valid_loss: float,
        current_valid_f1: float,
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
            epoch (int): the current epoch.
            model (nn.Module): the trained model.
            optimizer (torch.optim): the optimizer objet.
            fold (Union[int, None]): the current fold.
        """
        if current_valid_f1 > self.best_valid_f1:
            self.best_valid_loss = current_valid_loss
            self.best_valid_f1 = current_valid_f1
            self.best_epoch = epoch
            self.save_model = True
        
        if self.save_model:
            print("\nSaving model...")
            print(f"Epoch: {epoch}")
            print(f"Validation F1-Score: {current_valid_f1:1.6f}")
            print(f"Validation Loss: {current_valid_loss:1.6f}\n")
            
            if not fold is None:
                path = os.path.join(self.output_dir, f"{self.model_name}_fold{fold}.pth")
            else:
                path = os.path.join(self.output_dir, f"{self.model_name}.pth")
                
            torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                path,
            )
            self.save_model = False