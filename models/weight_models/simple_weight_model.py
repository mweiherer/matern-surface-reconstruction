import torch.nn as nn

from models.networks.layers import ResnetBlockFC


class SimpleWeightModel(nn.Module):
    ''' 
    This class implements the standard weight model as introduced in the NKF paper. It 
    predicts per-point weights to make solutions more robust against noise by solving the 
    weighted kernel ridge regression problem.
    :param cfg: The config file
    '''
    def __init__(self, cfg):
        super(SimpleWeightModel, self).__init__()

        feature_dim = cfg['model']['feature_field']['feature_dim']
        self.n_blocks = cfg['model']['weight_model']['n_blocks']
        hidden_dim = cfg['model']['weight_model']['hidden_dim']

        self.actvn = nn.ReLU()
        self.final_actvn = nn.Sigmoid()

        self.fc_p = nn.Linear(in_features = 3, out_features = hidden_dim)

        self.fc_c = nn.ModuleList([
            nn.Linear(feature_dim, hidden_dim) for _ in range(self.n_blocks)
        ])

        self.blocks = nn.ModuleList([
            ResnetBlockFC(size_in = hidden_dim) for _ in range(self.n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_dim, out_features = 1)


    def forward(self, points, labels):
        ''' 
        Forward function of SimpleWeightModel.
        :param points: torch.Tensor of size [b, 2n, 3], where n is number of observations
        :param labels: torch.Tensor of size [b, 2n, 1]
        :return: torch.Tensor of size [b, 2n]
        '''
        net = self.fc_p(points)
        for i in range(self.n_blocks):
            net += self.fc_c[i](labels)
            net = self.blocks[i](net)
        out = self.fc_out(self.actvn(net)).squeeze(dim = -1)

        return self.final_actvn(out) * 2.0 # To scale to [0, 2].