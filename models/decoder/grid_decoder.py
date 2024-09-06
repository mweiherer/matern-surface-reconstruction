import torch
import torch.nn as nn
import torch.nn.functional as F


class GridDecoder(nn.Module):
    def __init__(self, cfg):
        super(GridDecoder, self).__init__()
        decoder_cfg = cfg['model']['feature_field']['decoder']

        self.normalize = decoder_cfg['normalize']

    def forward(self, query_points, feature_grid):
        point_features = F.grid_sample(feature_grid, query_points[:, :, None, None], padding_mode = 'border',
                                       align_corners = True, mode = 'bilinear').squeeze(dim = -1).squeeze(dim = -1).transpose(1, 2)
        
        if self.normalize:
            feat_norm = torch.maximum(torch.norm(point_features.clone(), dim = -1, keepdim = True), torch.tensor([1e-7]).to(point_features))
            point_features /= feat_norm

        return point_features