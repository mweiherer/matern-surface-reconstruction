import torch
import numpy as np

from models.kernels.base_kernel import BaseKernel


class MaternKernel(BaseKernel):
    '''
    This class implements the Matérn kernel family. Specifically, we implement the 
    Matérn 1/2 (also known as Laplace or Exponential kernel), Matérn 3/2, Matérn 5/2, 
    and the limiting kernel (as nu approaches infinity), which is the Gaussian kernel.
    Please see https://en.wikipedia.org/wiki/Matérn_covariance_function.
    :param cfg: The config file
    '''
    def __init__(self, cfg):
        super(MaternKernel, self).__init__(cfg['model']['kernel']['reg_weight'])

        self.order = cfg['model']['kernel']['kwargs']['order']
        self.h = cfg['model']['kernel']['kwargs']['h']

    def _compute_gram_matrix(self, x, y):
        d_xy = torch.cdist(x, y)

        if self.order == '1/2':
            return torch.exp(-d_xy / self.h)
        
        if self.order == '3/2':
            return torch.exp(-(np.sqrt(3) * d_xy) / self.h) * (1 + (np.sqrt(3) * d_xy) / self.h) 
       
        if self.order == '5/2':
            return torch.exp(-(np.sqrt(5) * d_xy) / self.h) * (1 + (np.sqrt(5) * d_xy) / self.h + (5 * d_xy ** 2) / (3 * self.h ** 2))
        
        if self.order == 'inf':
            return torch.exp(-d_xy ** 2 / (2 * self.h ** 2))
 
    def solve(self, x, y, weights):
        if self.result is not None:
            raise RuntimeError('Please first call reset() before solve().')

        G = self._compute_gram_matrix(x, x)

        super().solve(G, x, y, weights)

    def evaluate(self, x_query):
        if self.result is None:
            raise RuntimeError('Can not call evaluate() before solve().')

        G = self._compute_gram_matrix(x_query, self.result['x'])

        return super().evaluate(G)