import torch.nn as nn


class FeatureField(nn.Module):
    def __init__(self, encoder, decoder):
        super(FeatureField, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.input_encoding = None

    def reset(self):
        self.input_encoding = None

    def forward(self, input_points, input_features):
        self.encode(input_points, input_features)
        return self.evaluate(input_points)

    def encode(self, input_points, input_features = None):
        ''' 
        Encodes input points. Optionally conditions input encoding on per-point features.
        :param input_points: torch.Tensor of size [b, n, 3], where n is number of observations
        :param input_features: torch.Tensor of size [b, n, d], where d is input feature dimension
        '''
        if self.input_encoding is not None:
            raise RuntimeError('Can not call encode() before reset().')
        
        self.input_encoding = self.encoder(input_points, input_features)

    def evaluate(self, query_points):
        '''
        Evaluates input encoding at any given points.
        :param query_points: torch.Tensor of size [b, n, 3]
        :return point_features: torch.Tensor of size [b, n, q], where q is target feature dimension
        '''
        if self.input_encoding is None:
            raise RuntimeError('Can not call evaluate() before encode().')
        
        return self.decoder(query_points, self.input_encoding)