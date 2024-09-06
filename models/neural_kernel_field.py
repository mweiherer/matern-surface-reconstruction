import torch
import torch.nn as nn


class NeuralKernelField(nn.Module):
    ''' 
    This class implements a Neural Kernel Field (NKF) as introduced in:
        Williams et al., Neural Fields as Learnable Kernels for 3D Reconstruction, CVPR'22.
    :param feature_field: Feature field used to predict per-point features
    :param kernel: The kernel to be used
    :param weight_model: Optional model predicting per-point weights for weighted kernel ridge regression
    :param cfg: The config file
    '''
    def __init__(self, feature_field, kernel, weight_model, cfg):
        super(NeuralKernelField, self).__init__()
        self.feature_field = feature_field
        self.kernel = kernel
        self.weight_model = weight_model
        
        self.input_type = cfg['model']['kernel']['input_type']
        
        # Cache things for logging.
        self.support_features = None

    def reset(self):
        self.feature_field.reset()
        self.kernel.reset()
        self.support_features = None

    def forward(self, observations, query_points, chunk_size = -1):
        obs_points, obs_labels, support_points, support_labels = observations
        self.fit(obs_points, obs_labels, support_points, support_labels)
        return self.predict(query_points, chunk_size)
    
    def _prepare_input(self, features, points):
        if self.input_type == 'concat':
            return torch.cat([features, points], dim = -1)
        if self.input_type == 'only_points':
            return points
        if self.input_type == 'only_features':
            return features

        raise ValueError("Invalid input type. Must be one of 'concat', 'only_points', 'only_features'.")
        
    def fit(self, obs_points, obs_labels, support_points, support_labels):
        ''' 
        Solves kernel regression according to Eq. 9 of paper.
        :param obs_points: torch.Tensor of size [b, 2n, 3]
        :param obs_labels: torch.Tensor of size [b, 2n, 1]
        :param support_points: torch.Tensor of size [b, 2n, 3]
        :param support_labels: torch.Tensor of size [b, 2n, 1]
        '''
        self.feature_field.encode(obs_points, obs_labels)
        self.support_features = self.feature_field.evaluate(support_points)
    
        inputs = self._prepare_input(self.support_features, support_points)

        if self.weight_model is not None:
            weights = self.weight_model(support_points, self.support_features)
        else:
            weights = None
    
        self.kernel.solve(inputs, support_labels, weights)

    def _predict_chunk(self, query_points):
        query_feat = self.feature_field.evaluate(query_points)

        inputs = self._prepare_input(query_feat, query_points)
    
        return self.kernel.evaluate(inputs)

    def predict(self, query_points, chunk_size = -1):
        ''' 
        Computes function value at given points according to Eq. 10 of paper. 
        Computation can optionally be chunked by supplying a chunk size.
        :param query_pts: torch.Tensor of size [b, m, 3]
        :param chunk_size: Optional chunk size greater than zero
        :return fx: torch.Tensor of size [b, m, 1]
        '''
        if chunk_size > 0:
            chunks = torch.split(query_points, chunk_size, dim = 1)
            return torch.cat([self._predict_chunk(pts) for pts in chunks], dim = 1)

        return self._predict_chunk(query_points)