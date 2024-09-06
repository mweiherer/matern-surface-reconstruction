import torch
import torch.nn as nn
from torch_scatter import scatter_max

from models.networks.layers import ResnetBlockFC


# Code adapted from: https://github.com/autonomousvision/convolutional_occupancy_networks. 
class LocalPoolPointNet(nn.Module):
    '''
    This class implements a ResNet-based PointNet encoder with local pooling as proposed in
        Peng et al., Convolutional Occupancy Networks, ECCV'20.
    :param input_dim: Dimension of the input points
    :param output_dim: Feature dimension
    :param hidden_dim: Hidden dimension
    :param n_blocks: Number of ResNet blocks
    :param grid_resolution: Resolution of the feature grid
    '''
    def __init__(self, input_dim, output_dim, hidden_dim, n_blocks, grid_resolution):
        super(LocalPoolPointNet, self).__init__()

        self.fc_pos = nn.Linear(input_dim, 2 * hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2 * hidden_dim, hidden_dim) for _ in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, output_dim)
        
        self.feature_dim = output_dim
        self.grid_resolution = grid_resolution

    def pool_local(self, index, c): 
        c_out = scatter_max(c.permute(0, 2, 1), index, dim_size = self.grid_resolution ** 3)[0]
        c_out = c_out.gather(dim = 2, index = index.expand(-1, c.size(2), -1))
        return c_out.permute(0, 2, 1)
    
    def generate_grid_features(self, p, index, c):
        fea_grid = c.new_zeros(p.size(0), self.feature_dim, self.grid_resolution ** 3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_max(c, index, out = fea_grid)[0]
        fea_grid = fea_grid.reshape(p.size(0), self.feature_dim, 
                                    self.grid_resolution, self.grid_resolution, self.grid_resolution)
        return fea_grid

    @staticmethod
    def normalize_3d_coordinate(p, padding = 0):
        '''
        Normalizes points living in [-1, 1]^3 to [0, 2)^3. 
        :param p: Input points as torch.Tensor of size [b, n, 3]
        :return p_norm: Normalized input points
        '''
        if padding > 0: p_norm = p / (1 + padding + 10e-4) # (-1, 1)
        p_norm = p + 1 # (0, 2)

        if p_norm.max() >= 2: p_norm[p_norm >= 2] = 2 - 10e-5
        if p_norm.min() < 0: p_norm[p_norm < 0] = 0.0

        return p_norm
    
    def coordinates2index(self, x):
        x = (x / 2 * self.grid_resolution).long()
        index = x[:, :, 0] + self.grid_resolution * (x[:, :, 1] + self.grid_resolution * x[:, :, 2])
        return index[:, None, :]

    def forward(self, p, p_feat = None):
        '''
        Forwards points (and optionally, per-point features) through the network. Important: Points 
        are assumed to live in [-1, 1]^3 and should be centered at the origin, (0, 0, 0).
        :param p: Points as torch.Tensor of size [b, n , 3]
        :param p_feat: Optional, per-point features as torch.Tensor of size [b, n, d]
        :return: Feature grid as torch.Tensor of size [b, feature_dim, feature_dim, feature_dim, feature_dim]
        '''
        input = torch.cat([p, p_feat], dim = -1) if p_feat is not None else p
 
        p_norm = self.normalize_3d_coordinate(p, padding = 0.1) # Normalize to [0, 2)^3.
        index = self.coordinates2index(p_norm)

        net = self.fc_pos(input)
        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(index, net)
            net = torch.cat([net, pooled], dim = 2)
            net = block(net)
        c = self.fc_c(net)

        return self.generate_grid_features(p, index, c)