import torch
import numpy as np
import os
import yaml


# Code adapted from https://github.com/autonomousvision/convolutional_occupancy_networks/blob/master/src/data/core.py.
class ShapeNetDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, transform):
        super(ShapeNetDataset, self).__init__()

        if mode not in ['train', 'val', 'test']:
            raise ValueError(f"Unknown mode {mode}. Please choose either 'train', 'val', or 'test'.")
        
        self.dataset_root = cfg['data']['dataset_root']

        categories = os.listdir(self.dataset_root)
        categories = [c for c in categories
                      if os.path.isdir(os.path.join(self.dataset_root, c))]

        # Read metadata file.
        metadata_file = os.path.join(self.dataset_root, 'metadata.yaml')

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f'Metadata file {metadata_file} not found.')
        
        # Get all models.
        self.models = []
        for c_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = c_idx # Set index.
         
            split_file = os.path.join(self.dataset_root, c, mode + '.lst')

            with open(split_file, 'r') as f:
                models_c = f.read().split('\n')
                
            if '' in models_c:
                models_c.remove('')

            self.models += [
                {'category': c, 'model': m}
                for m in models_c
            ]

        self.dtype = np.float64

        self.transform = transform
        self.translate, self.scale = (0.0, 2.0) # To scale from [-0.5, 0.5]^3 to [-1, 1]^3.

    def __len__(self):
        return len(self.models)
 
    def __getitem__(self, idx):
        category = self.models[idx]['category']
        model = self.models[idx]['model']
        c_idx = self.metadata[category]['idx']
        
        model_path = os.path.join(self.dataset_root, category, model)

        surface_points_dict = np.load(os.path.join(model_path, 'pointcloud.npz'))
        volume_points_dict = np.load(os.path.join(model_path, 'points.npz'))

        surf_pts = surface_points_dict['points'].astype(self.dtype) # Raw surf_pts are in [-0.5, 0.5]^3.

        # surf_pts should technically be scaled to [-0.5, 0.5]^3, but in practice, we found that sometimes
        # they exeed this range. To scale them to [-1, 1]^3 exactly, we have to compute a custom scaling 
        # factor for each shape, which is less than 2 (the anticipated scaling factor).
        actual_scale = 1.0 / np.maximum(np.abs(surf_pts.min()), np.abs(surf_pts.max()))

        ret_dict = {'shape_id': model, 'scale': actual_scale}

        ret_dict['surf_points'] = actual_scale * (torch.from_numpy(surf_pts) + self.translate)

        surf_nms = surface_points_dict['normals'].astype(self.dtype)
        ret_dict['surf_normals'] = torch.from_numpy(surf_nms)

        vol_pts = volume_points_dict['points'] # Raw vol_pts are in [-0.55, 0.55]^3.
        
        # To break symmetry, see https://github.com/autonomousvision/convolutional_occupancy_networks/blob/master/src/data/fields.py. 
        if vol_pts.dtype == np.float16:
            vol_pts = vol_pts.astype(self.dtype)
            vol_pts += 1e-4 * np.random.randn(*vol_pts.shape)

        ret_dict['vol_points'] = actual_scale * (torch.from_numpy(vol_pts) + self.translate)

        vol_occ = -2.0 * (
                np.unpackbits(volume_points_dict['occupancies']).astype(self.dtype) - 0.5)
        ret_dict['vol_occs'] = torch.from_numpy(vol_occ)

        if self.transform is not None:
            return self.transform(ret_dict)
    
        return ret_dict