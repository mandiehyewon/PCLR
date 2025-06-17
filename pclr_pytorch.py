import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict

from tensorflow.keras.models import load_model


class ResidualBlock(nn.Module):
    def __init__(self, downsample, in_channels, out_channels, kernel_size=16, stride=1, padding="same"):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, downsample, padding=0, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.kernel_size = kernel_size
        self.downsample = downsample
        self.pool = nn.MaxPool1d(kernel_size=downsample, stride=downsample)
        self.identity_layer = None
        if in_channels != out_channels:
            self.identity_layer = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

        self.manual_padding = ((self.kernel_size + 1 - self.downsample) // 2, (self.kernel_size - self.downsample) // 2)

    def forward(self, x, residual):
        residual = self.pool(residual)
        residual = residual if self.identity_layer is None else self.identity_layer(residual)
        out = F.relu(self.bn1(self.conv1(x)))
        # Manual same padding, Pytorch same padding doesn't work with stride > 1
        out = self.conv2(F.pad(out, self.manual_padding))
        # Add skip before bn&act https://github.com/broadinstitute/ml4h/blob/master/model_zoo/PCLR/build_model.py#L249
        #   TODO: make preactivation into an option
        out += residual
        residual = out
        out = F.relu(self.bn2(out))
        return out, residual


class ResNet1D(nn.Module):
    def __init__(self, config):
        super(ResNet1D, self).__init__()
        self.config = config
        self.conv1 = nn.Conv1d(self.config.num_input_channels, self.config.num_filters[0], kernel_size=16, padding="same", bias=False)
        self.bn1 = nn.BatchNorm1d(self.config.num_filters[0])

        self.num_res_blocks = len(self.config.num_filters) - 1  # last one is representation dim
        res_blocks = []
        for block in range(self.num_res_blocks):
            res_blocks.append(ResidualBlock(self.config.pool_kernel_size_res_block, self.config.num_filters[block], self.config.num_filters[block + 1]))
        self.res_blocks = nn.ModuleList(res_blocks)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        if self.config.pretrained_weights is not None:
            self.load_pretrained_keras_weights(self.config.pretrained_weights)

    def load_pretrained_keras_weights(self, file):
        keras_model = load_model(file)

        with torch.no_grad():
            self.conv1.weight.copy_(
                torch.tensor(
                    keras_model.layers[1].get_weights()[0][:, :self.config.num_input_channels]
                ).permute(2, 1, 0)
            )
            self.bn1.weight.copy_(torch.tensor(keras_model.layers[2].get_weights()[0]))

            self.res_blocks[0].conv1.weight.copy_(
                torch.tensor(keras_model.layers[4].get_weights()[0]).permute(2, 1, 0))
            self.res_blocks[0].bn1.weight.copy_(torch.tensor(keras_model.layers[5].get_weights()[0]))
            self.res_blocks[0].conv2.weight.copy_(
                torch.tensor(keras_model.layers[8].get_weights()[0]).permute(2, 1, 0))
            self.res_blocks[0].bn2.weight.copy_(torch.tensor(keras_model.layers[11].get_weights()[0]))
            self.res_blocks[0].identity_layer.weight.copy_(
                torch.tensor(keras_model.layers[9].get_weights()[0]).permute(2, 1, 0))

            self.res_blocks[1].conv1.weight.copy_(
                torch.tensor(keras_model.layers[13].get_weights()[0]).permute(2, 1, 0))
            self.res_blocks[1].bn1.weight.copy_(torch.tensor(keras_model.layers[14].get_weights()[0]))
            self.res_blocks[1].conv2.weight.copy_(
                torch.tensor(keras_model.layers[17].get_weights()[0]).permute(2, 1, 0))
            self.res_blocks[1].bn2.weight.copy_(torch.tensor(keras_model.layers[20].get_weights()[0]))
            self.res_blocks[1].identity_layer.weight.copy_(
                torch.tensor(keras_model.layers[18].get_weights()[0]).permute(2, 1, 0))

            self.res_blocks[2].conv1.weight.copy_(
                torch.tensor(keras_model.layers[22].get_weights()[0]).permute(2, 1, 0))
            self.res_blocks[2].bn1.weight.copy_(torch.tensor(keras_model.layers[23].get_weights()[0]))
            self.res_blocks[2].conv2.weight.copy_(
                torch.tensor(keras_model.layers[26].get_weights()[0]).permute(2, 1, 0))
            self.res_blocks[2].bn2.weight.copy_(torch.tensor(keras_model.layers[29].get_weights()[0]))
            self.res_blocks[2].identity_layer.weight.copy_(
                torch.tensor(keras_model.layers[27].get_weights()[0]).permute(2, 1, 0))

            self.res_blocks[3].conv1.weight.copy_(
                torch.tensor(keras_model.layers[31].get_weights()[0]).permute(2, 1, 0))
            self.res_blocks[3].bn1.weight.copy_(torch.tensor(keras_model.layers[32].get_weights()[0]))
            self.res_blocks[3].conv2.weight.copy_(
                torch.tensor(keras_model.layers[35].get_weights()[0]).permute(2, 1, 0))
            self.res_blocks[3].bn2.weight.copy_(torch.tensor(keras_model.layers[38].get_weights()[0]))
            self.res_blocks[3].identity_layer.weight.copy_(
                torch.tensor(keras_model.layers[36].get_weights()[0]).permute(2, 1, 0))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        residual = x
        for res_block in self.res_blocks:
            x, residual = res_block(x, residual)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        return x


if __name__ == "__main__":
    config = EasyDict({
            "num_input_channels": 1,  # number of ECG leads (e.g. 1 for lead i, 12 for all 12 leads)
            "num_filters": [64, 128, 196, 256, 320],  # these are PCLR dimensions
            "pool_kernel_size_res_block": 4,
            "pretrained_weights": "./PCLR.h5"
    })
    model = ResNet1D(config)
    # torch.save(model.state_dict(), "./PCLR_pytorch.pt")
