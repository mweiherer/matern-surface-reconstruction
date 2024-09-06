import torch
from torch.nn import BCEWithLogitsLoss


def sdf_multitask_loss(fx_free, fx_surf, free_occ):
    '''
    Computes a multitask loss to learn an SDF and its sign (ie, occupancies) simultaneously.
    This was proposed in
        Sitzmann et al., MetaSDF: Meta-learning Signed Distance Functions, NeurIPS'20.
    :param fx_free: Computed SDF in the volume
    :param fx_surf: Computed SDF on surface
    :param free_occ: Ground-truth occupancies for points in the volume
    :return: Dictionary containing BCE-based freespace loss and L1 surface loss
    '''
    batch_size = free_occ.shape[0]  
 
    freespace_loss = BCEWithLogitsLoss(reduction = 'mean')(fx_free, 0.5 * free_occ + 0.5) / batch_size    
    surface_loss = torch.abs(fx_surf).mean()

    return {
        'freespace_loss': freespace_loss,
        'surface_loss': surface_loss
    }