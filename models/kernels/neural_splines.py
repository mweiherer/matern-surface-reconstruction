import torch
import numpy as np

from models.kernels.base_kernel import BaseKernel
from utils.stable_angle import stable_angle


class NeuralSplinesKernel(BaseKernel):
    '''
    This class implements the Neural Splines kernel as introduced in:
        Williams et al., Neural Splines: Fitting 3D Surfaces with Infinitely-Wide Neural Networks, CVPR'21.
    We are using the formulation that arises from a Gaussian initialization, see Prop. 7 of Appendix.
    :param cfg: The config file
    '''
    def __init__(self, cfg):
        super(NeuralSplinesKernel, self).__init__(cfg['model']['kernel']['reg_weight'])
        self.input_dim = 3

    def _compute_gram_matrix(self, x, y):
        x = self._cat_ones(x)
        y = self._cat_ones(y)

        # Computes the Neural Splines kernel for *unbatched* inputs.
        def neural_splines(x, y):
            angle = stable_angle(x, y)
            norm_x = torch.norm(x.unsqueeze(1), dim = -1)
            norm_y = torch.norm(y.unsqueeze(0), dim = -1)
            return norm_x * norm_y * (torch.sin(angle) + 2.0 *
                                     (np.pi - angle) * torch.cos(angle)) / np.pi

        K = [neural_splines(x[i, ...], y[i, ...]) for i in range(x.shape[0])]
        return torch.stack(K)

    def _cat_ones(self, x):
        if x.shape[-1] != (self.input_dim + 1): # Check if 1 already concatenated.
            return torch.cat([x, torch.ones(x.shape[0], x.shape[1], 1).to(x)], dim = -1)
        return x

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