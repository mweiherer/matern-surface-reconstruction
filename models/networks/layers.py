import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial


# Source: https://github.com/autonomousvision/occupancy_networks/blob/master/im2mesh/layers.py. 
class ResnetBlockFC(nn.Module):
    def __init__(self, size_in, size_out = None, size_h = None):
        super(ResnetBlockFC, self).__init__()

        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias = False)

        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx

# Code adapted from: https://github.com/autonomousvision/convolutional_occupancy_networks/blob/master/src/encoder/unet3d.py.
class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, apply_pooling, num_groups):
        super(UNetEncoder, self).__init__()

        if apply_pooling:
            self.pooling = nn.MaxPool3d(kernel_size = (2, 2, 2))
        else:
            self.pooling = None

        conv1_out_channels = out_channels // 2 if (out_channels // 2 > in_channels) else in_channels

        self.gn_1 = nn.GroupNorm(num_groups = num_groups, num_channels = in_channels)
        self.conv_1 = nn.Conv3d(in_channels, conv1_out_channels, kernel_size = 3, padding = 1, bias = False)

        self.gn_2 = nn.GroupNorm(num_groups = num_groups, num_channels = conv1_out_channels)
        self.conv_2 = nn.Conv3d(in_channels, out_channels, kernel_size = 3, padding = 1, bias = False)

        self.actvn_fn = nn.ReLU()

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)

        x = self.gn_1(x)
        x = self.conv_1(x)
        x = self.actvn_fn(x)

        x = self.gn_2(x)
        x = self.conv_2(x)
        x = self.actvn_fn(x)

        return x

# Code adapted from: https://github.com/autonomousvision/convolutional_occupancy_networks/blob/master/src/encoder/unet3d.py.
class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups):
        super(UNetDecoder, self).__init__()

        self.upsampling = Upsampling(in_channels = in_channels, out_channels = out_channels)

        self.gn_1 = nn.GroupNorm(num_groups = num_groups, num_channels = in_channels)
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size = 3, padding = 1, bias = False)

        self.gn_2 = nn.GroupNorm(num_groups = num_groups, num_channels = out_channels)
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size = 3, padding = 1, bias = False)

        self.actvn_fn = nn.ReLU()

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features = encoder_features, x = x)
        x = torch.cat((encoder_features, x), dim = 1)

        x = self.gn_1(x)
        x = self.conv_1(x)
        x = self.actvn_fn(x)

        x = self.gn_2(x)
        x = self.conv_2(x)
        x = self.actvn_fn(x)

        return x

# Source: https://github.com/autonomousvision/convolutional_occupancy_networks/blob/master/src/encoder/unet3d.py.
class Upsampling(nn.Module):
    def __init__(self, transposed_conv = False, in_channels = None, out_channels = None, kernel_size = 3,
                 scale_factor = (2, 2, 2), mode = 'nearest'):
        super(Upsampling, self).__init__()

        if transposed_conv:
            self.upsample = nn.ConvTranspose3d(in_channels,
                                               out_channels,
                                               kernel_size = kernel_size,
                                               stride = scale_factor,
                                               padding = 1)
        else:
            self.upsample = partial(self._interpolate, mode = mode)

    def forward(self, encoder_features, x):
        output_size = encoder_features.size()[2:]
        return self.upsample(x, output_size)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size = size, mode = mode)