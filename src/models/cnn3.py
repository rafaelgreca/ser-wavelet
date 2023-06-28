import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

def init_layer(layer: nn.Module) -> None:
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
def init_bn(bn: nn.Module) -> None:
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x
    
class CNN3_Mode2(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv_block1 = ConvBlock(in_channels=5, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, 3, bias=True)
        
        self.init_weight()

    def init_weight(self) -> None:
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(
        self,
        input: torch.Tensor
    ) -> torch.Tensor:
        """
        Input: (batch_size, data_length)
        """
        x = self.conv_block1(input, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5)
        x = F.relu_(self.fc1(x))
        output = self.fc_audioset(x)

        return output
    
class CNN10(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        sample_rate = 32000
        window_size = 1024
        hop_size = 320
        mel_bins = 64
        fmin = 0
        fmax = 32000
        
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)
        
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, 527, bias=True)
        
        self.init_weight()

    def init_weight(self) -> None:
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(
        self,
        input: torch.Tensor
    ) -> torch.Tensor:
        x = self.conv_block1(input, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5)
        x = F.relu_(self.fc1(x))
        
        embedding = F.dropout(x, p=0.5)
        output_dict = {"embedding": embedding}

        return output_dict
    
class Transfer_CNN10(nn.Module):
    def __init__(self, load_pretrained, freeze_base):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super().__init__()
        
        self.base = CNN10()

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(512, 3, bias=True)

        if load_pretrained:
            path = "./pretrained/Cnn10_mAP=0.380.pth"
            self.load_from_pretrain(
                pretrained_checkpoint_path=path
            )
            print(f"\nPretrained model loaded from path {path}\n")
            
        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input)
        embedding = output_dict['embedding']
        clipwise_output = self.fc_transfer(embedding)
        output_dict['clipwise_output'] = clipwise_output
 
        return clipwise_output