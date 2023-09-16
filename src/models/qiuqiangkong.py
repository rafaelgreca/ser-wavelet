import torch
import torch.nn as nn
import torch.nn.functional as F


def init_layer(layer: nn.Module) -> None:
    """
    Initialize a Linear or Convolutional layer.
    """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn: nn.Module) -> None:
    """
    Initialize a Batchnorm layer.
    """
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x


class ConvBlock5x5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock5x5, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x


class CNN10(nn.Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.conv_block1 = ConvBlock(in_channels=input_channels, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, 527, bias=True)

        self.init_weight()

    def init_weight(self) -> None:
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(input, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        output_dict = {"embedding": x}

        return output_dict


class Cnn6(nn.Module):
    def __init__(self, input_channels: int):
        super().__init__()
        self.conv_block1 = ConvBlock5x5(in_channels=input_channels, out_channels=64)
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, 527, bias=True)

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)
        """
        x = self.conv_block1(input, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        output_dict = output_dict = {"embedding": x}

        return output_dict


class Transfer_CNN6(nn.Module):
    def __init__(
        self,
        input_channels: int,
        load_pretrained: bool,
        freeze_base: bool,
        num_classes: int,
        pretrained_model_path: str = "./pretrained/Cnn6_mAP=0.343.pth",
    ) -> None:
        """
        Classifier for a new task using pretrained Cnn6 as a sub module.
        """
        super().__init__()

        self.base = Cnn6(input_channels)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(512, num_classes, bias=True)

        if load_pretrained:
            self.load_from_pretrain(pretrained_checkpoint_path=pretrained_model_path)
            print(f"\nPretrained model loaded from path {pretrained_model_path}\n")

        if freeze_base:
            print(f"\nFreezing pretrained layers weights\n")

            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint["model"])

    def forward(self, input, mixup_lambda=None):
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict["embedding"]
        clipwise_output = self.fc_transfer(embedding)
        output_dict["clipwise_output"] = clipwise_output

        return clipwise_output


class Transfer_CNN10(nn.Module):
    def __init__(
        self,
        input_channels: int,
        load_pretrained: bool,
        freeze_base: bool,
        num_classes: int,
        pretrained_model_path: str = "./pretrained/Cnn10_mAP=0.380.pth",
    ) -> None:
        """
        Classifier for a new task using pretrained Cnn10 as a sub module.
        """
        super().__init__()

        self.base = CNN10(input_channels)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(512, num_classes, bias=True)

        if load_pretrained:
            self.load_from_pretrain(pretrained_checkpoint_path=pretrained_model_path)
            print(f"\nPretrained model loaded from path {pretrained_model_path}\n")

        if freeze_base:
            print(f"\nFreezing pretrained layers weights\n")

            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint["model"])

    def forward(self, input):
        output_dict = self.base(input)
        embedding = output_dict["embedding"]
        clipwise_output = self.fc_transfer(embedding)
        output_dict["clipwise_output"] = clipwise_output

        return clipwise_output
