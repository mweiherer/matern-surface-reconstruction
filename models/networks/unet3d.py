import torch.nn as nn

from models.networks.layers import UNetDecoder, UNetEncoder


# Code adapted from: https://github.com/autonomousvision/convolutional_occupancy_networks/blob/master/src/encoder/unet3d.py.
class UNet3D(nn.Module):
    def __init__(self, in_channels, f_maps, num_levels, num_groups):
        super(UNet3D, self).__init__()
        out_channels = in_channels
        
        f_maps = [f_maps * 2 ** k for k in range(num_levels)]

        # Create encoder.
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = UNetEncoder(in_channels,
                                      out_feature_num,
                                      apply_pooling = False,
                                      num_groups = num_groups)
            else:
                encoder = UNetEncoder(f_maps[i - 1],
                                      out_feature_num,
                                      apply_pooling = True,
                                      num_groups = num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # Create decoder.
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
           
            decoder = UNetDecoder(in_feature_num,
                                  out_feature_num,
                                  num_groups = num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # Final convolution layer.
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

    def forward(self, x):
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)

        encoders_features = encoders_features[1:]

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)

        return self.final_conv(x)