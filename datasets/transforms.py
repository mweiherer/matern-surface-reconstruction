import torch

from utils.common import offset_along_normal


class ShapeReconstruction:
    def __init__(self, cfg):
        data_dict = cfg['data']

        self.num_observations = data_dict['num_observations']
        self.observation_noise = data_dict['observation_noise']
        self.num_surface_points = data_dict['num_surface_points']
        self.num_freespace_points = data_dict['num_freespace_points'] 
       
        self.num_eval_points = cfg['eval']['num_eval_points']
        self.surf_eps = cfg['misc']['surf_eps']

    def sample_supervision_points(self, vol_points, vol_occs, surf_points, surf_normals):
        rand_surf_indc = torch.randperm(surf_points.shape[0])[:self.num_surface_points]
        surf_points, surf_normals = surf_points[rand_surf_indc, :], surf_normals[rand_surf_indc, :]
        offset_points, offset_occs = offset_along_normal(surf_points, surf_normals, self.surf_eps)
    
        rand_vol_indc = torch.randperm(vol_points.shape[0])[:self.num_freespace_points]

        return {
            'vol_points': torch.cat([vol_points[rand_vol_indc, :], offset_points], dim = 0),
            'vol_occs': torch.cat([vol_occs[rand_vol_indc, :], offset_occs], dim = 0),
            'surf_points': surf_points,
            'surf_normals': surf_normals
        }

    def sample_eval_points(self, vol_points, vol_occs, surf_points, surf_normals):
        rand_surf_indc = torch.randperm(surf_points.shape[0])[:self.num_eval_points]
        rand_vol_indc = torch.randperm(vol_points.shape[0])[:self.num_eval_points]
    
        return {
            'chamfer_points': surf_points[rand_surf_indc, :],
            'chamfer_normals': surf_normals[rand_surf_indc, :],
            'iou_points': vol_points[rand_vol_indc, :],
            'iou_occs': vol_occs[rand_vol_indc, :]
        }

    def __call__(self, data):
        shape_id, scale = data['shape_id'], data['scale']
        vol_points, vol_occs = data['vol_points'], data['vol_occs'].unsqueeze(dim = -1)
        surf_points, surf_normals = data['surf_points'], data['surf_normals']
        
        # Sample observation points (points to condition on) and add noise, if requested. 
        rand_indc = torch.randperm(surf_points.shape[0])[:self.num_observations]
        obs_points = surf_points[rand_indc, :]
        obs_normals = surf_normals[rand_indc, :]

        if self.observation_noise > 0.0:
            obs_points += torch.randn_like(obs_points) * self.observation_noise

        offset_points, offset_occs = offset_along_normal(obs_points, obs_normals, self.surf_eps)

        return_dict = {
            'shape_id': shape_id,
            'scale': scale,
            'obs_points': offset_points,
            'obs_normals': obs_normals,
            'obs_occs': offset_occs * 0.5,
            'supp_points': offset_points,
            'supp_occs': offset_occs
        }
       
        # Sample supervision points (on-surface and volume points) used to compute the loss.
        return_dict.update(self.sample_supervision_points(vol_points, vol_occs, surf_points, surf_normals))
        
        # Sample points to compute evaluation metrics.
        return_dict.update(self.sample_eval_points(vol_points, vol_occs, surf_points, surf_normals))

        return return_dict