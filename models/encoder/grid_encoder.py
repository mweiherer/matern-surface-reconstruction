import torch.nn as nn

from models.networks import LocalPoolPointNet, UNet3D


class GridEncoder(nn.Module):
    def __init__(self, cfg):
        super(GridEncoder, self).__init__()
        encoder_cfg = cfg['model']['feature_field']['encoder']
        
        # PointNet parameters.
        input_dim = encoder_cfg['input_dim']
        output_dim = encoder_cfg['output_dim']
        hidden_dim = encoder_cfg['hidden_dim']
        n_blocks = encoder_cfg['n_blocks']
        grid_resolution = encoder_cfg['grid_resolution']

        # UNet parameters.
        f_maps = encoder_cfg['unet']['f_maps']
        num_levels = encoder_cfg['unet']['num_levels']
        num_groups = encoder_cfg['unet']['num_groups']

        self.local_pool_point_net = LocalPoolPointNet(input_dim, output_dim, hidden_dim, n_blocks, grid_resolution)
        self.unet3d = UNet3D(output_dim, f_maps, num_levels, num_groups)

    def forward(self, p, p_feat = None):
        input_encoding = self.local_pool_point_net(p, p_feat)
        return self.unet3d(input_encoding)