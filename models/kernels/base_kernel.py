import torch
import torch.nn as nn
from abc import ABC


class BaseKernel(nn.Module, ABC):
    '''
    Abstract base class for a kernel. Every newly defined kernel should inherit from this class 
    and override the _compute_gram_matrix() method. You don't have to touch the rest!
    :param reg_weight: The regularization weight used for solving the kernel ridge regression
    '''
    def __init__(self, reg_weight):
        super().__init__()
        self.reg_weight = reg_weight
        self.result = None

    def reset(self):
        self.result = None

    def _compute_gram_matrix(self, x, y):
        '''
        Computes the Gram matrix between d-dimensional n inputs x and m inputs y according to a kernel function.
        Should return a real, symmetric positive (semi-)definite n by m matrix (aka a Gram matrix).
        :param x: torch.Tensor of size [b, n, d]
        :param y: torch.Tensor of size [b, m, d] 
        '''
        raise NotImplementedError

    def solve(self, G, x, y, weights):
        '''
        Solves the kernel ridge regression or, in case weights are given, the weighted kernel
        ridge regression using Cholesky decomposition. The solution will be stored in the 'result'
        class variable. For more information, please see Eqs. 9 and 13 of the NKF paper:
            Williams et al., Neural Fields as Learnable Kernels for 3D Reconstruction, CVPR'22.
        :param G: The Gram matrix computed between all observations as torch.Tensor of size [b, 2n, 2n], where n is number of observations
        :param x: Observed points (with concat features) as torch.Tensor of size [b, 2n, d + 3], where d is feature dimension 
        :param y: Corresponding occupancies as torch.Tensor of size [b, 2n, 1]
        :param weights: Optional, per-point weights as torch.Tensor of size [b, 2n, 2n]
        '''
        # In case weights are given, solve weighted ridge regression.
        if weights is not None:
            G = weights.unsqueeze(dim = -1) * G * weights
            y = (y.squeeze(dim = -1) * weights).unsqueeze(dim = -1)

        A = G + self.reg_weight * torch.eye(G.shape[1]).repeat(G.shape[0], 1, 1).to(G)
   
        # Solve linear system of equations using Cholesky decomposition.
        L, _ = torch.linalg.cholesky_ex(A)
        alpha = torch.cholesky_solve(y, L)

        self.result = {'x': x, 'L': L, 'alpha': alpha}

    def evaluate(self, G):
        '''
        Evaluates the predicted function at m test points x (encoded in the Gram matrix, G). 
        For more information, please see Eq. 10 of the NKF paper.
        :param G: The Gram matrix computed between x and all observations as torch.Tensor of size [b, 2n, m]
        :return: The computed function value at x as torch.Tensor of size [b, m, 1]
        '''
        return G @ self.result['alpha']