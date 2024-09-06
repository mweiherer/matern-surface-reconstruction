import torch
import random
import numpy as np


def seed_everything(seed):
    if seed < 0:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def condition_numbers(L):
    '''
    Computes condition numbers for a batch of matrices based in its Cholesky decompositions.
    :param L: Cholesky decompositions as torch.Tensor of size b x n x n
    :return cond_numbers: Condition number for each matrix in the batch as torch.Tensor of size b
    '''
    eigvals = torch.diagonal(L, dim1 = -2, dim2 = -1)
    cond_numbers = (torch.max(eigvals, dim = 1)[0] / torch.min(eigvals, dim = 1)[0]) ** 2
    return cond_numbers

def offset_along_normal(points, normals, eps):
    offset_points = torch.cat([points + normals * eps,
                               points - normals * eps], dim = 0).to(points.dtype)
    offset_occs = torch.cat([torch.ones(points.shape[0]), 
                             -torch.ones(points.shape[0])]).to(points.dtype)
    return offset_points, offset_occs.unsqueeze(dim = -1)

# Source: https://stackoverflow.com/questions/29027792/get-average-value-from-list-of-dictionary.
def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = torch.mean(torch.Tensor([d[key] for d in dict_list]), dim = 0)
    return mean_dict