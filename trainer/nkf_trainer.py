import torch

from trainer import BaseTrainer
from utils.common import condition_numbers
from utils.eval import compute_evaluation_metrics, make_grid_points
from utils.loss_functions import sdf_multitask_loss


class NeuralKernelFieldTrainer(BaseTrainer):
    def __init__(self, model, optimizer, train_loader, eval_loader, test_loader, cfg):
        super(NeuralKernelFieldTrainer, self).__init__(model, optimizer, train_loader, eval_loader, test_loader, cfg)
        
        self.epochs_til_evaluation = self.cfg['train']['epochs_til_evaluation']
        self.accumulate_gradients = cfg['misc']['accumulate_grad']

    def train_step(self, batch_data):
        if self.accumulate_gradients:
            self.train_step_gradient_accumulation(batch_data)
        else:
            self.model.reset()

            self.optimizer.zero_grad()

            loss_dict = self.compute_loss(batch_data)

            # Check for numerical stability.
            cond_numbers = condition_numbers(self.model.kernel.result['L'])
            well_conditioned = False if any(cond_numbers > self.cfg['misc']['cond_threshold']) else True

            if not well_conditioned:
                raise Exception('Numerical instability. Skip batch.')

            total_loss = 0
            for weight, loss in zip(self.cfg['loss']['weights'], list(loss_dict.values())):
                total_loss += weight * loss

            total_loss.backward()  

            log_dict = {
                'loss_dict': {f'train/{key}': value for key, value in loss_dict.items()},
                'train/total_loss': total_loss,
                'train/cond_number': cond_numbers.mean(),
                'train/kernel_alpha_max': (self.model.kernel.result['alpha']).abs().max()
            }

            self.to_train_log(log_dict)

            self.optimizer.step()

    def train_step_gradient_accumulation(self, batch_data):
        self.optimizer.zero_grad()

        batch_loss_dict, batch_total_loss, batch_cond_numbers, batch_kernel_alpha_max = [],[],[],[]
        batch_size = batch_data['obs_points'].shape[0]

        for i in range(batch_size):
            self.model.reset()

            batch_data_i = {}
            for k, v in batch_data.items():
                batch_data_i[k] = [v[i]] if k == 'shape_id' else v[i, ...].unsqueeze(dim = 0)

            loss_dict_i = self.compute_loss(batch_data_i)

            # Check for numerical stability.
            cond_numbers_i = condition_numbers(self.model.kernel.result['L'])
            well_conditioned = False if any(cond_numbers_i > self.cfg['misc']['cond_threshold']) else True

            if not well_conditioned:
                print('Numerical instability. Skip shape.'); continue

            total_loss_i = 0
            for weight, loss in zip(self.cfg['loss']['weights'], list(loss_dict_i.values())):
                total_loss_i += weight * loss

            total_loss_i.backward()

            batch_loss_dict.append(loss_dict_i)
            batch_total_loss.append(total_loss_i)
            batch_cond_numbers.append(cond_numbers_i.mean())
            batch_kernel_alpha_max.append((self.model.kernel.result['alpha']).abs().max())

        log_dict = {
            'loss_dict': {f'train/{key}': value for key, value in batch_loss_dict.items()},
            'train/total_loss': torch.Tensor(batch_total_loss).mean(),
            'train/cond_number': torch.Tensor(batch_cond_numbers).mean(),
            'train/kernel_alpha_max': torch.Tensor(batch_kernel_alpha_max).max()
        }

        self.to_train_log(log_dict)

        self.optimizer.step()

    def compute_loss(self, batch_data):              
        obs_points = batch_data['obs_points'].to(self.device)
        supp_points = batch_data['supp_points'].to(self.device)
        obs_labels = batch_data['obs_occs'].to(self.device)
        supp_labels = batch_data['supp_occs'].to(self.device)

        # Points for supervision.
        surf_points = batch_data['surf_points'].to(self.device)
        free_points = batch_data['vol_points'].to(self.device)
        free_occ = batch_data['vol_occs'].to(self.device)

        observations = [obs_points, obs_labels, supp_points, supp_labels]
        query_points = torch.cat([free_points, surf_points], dim = -2)
        
        fx = self.model(observations, query_points)

        fx_free = fx[:, :free_points.shape[1], :]
        fx_surf = fx[:, free_points.shape[1]:, :]

        loss_dict = sdf_multitask_loss(fx_free, fx_surf, free_occ)
       
        return_dict = {}
        for objective in self.cfg['loss']['objectives']:
            return_dict[objective] = loss_dict[objective]

        return return_dict

    def eval_step(self, batch_data):
        self.model.reset()
  
        voxel_resolution = self.cfg['eval']['voxel_resolution']
        chunk_size = self.cfg['eval']['chunk_size']

        obs_points = batch_data['obs_points'].to(self.device)
        supp_points = batch_data['supp_points'].to(self.device)
        obs_labels = batch_data['obs_occs'].to(self.device)
        supp_labels = batch_data['supp_occs'].to(self.device)

        shape_ids = batch_data['shape_id']
        scales = batch_data['scale']

        # Points for evaluation.
        chamfer_points = batch_data['chamfer_points']
        chamfer_normals = batch_data['chamfer_normals']
        iou_points = batch_data['iou_points'].to(self.device)
        iou_occs = batch_data['iou_occs'].squeeze(dim = -1)

        grid_points = make_grid_points(voxel_resolution).repeat(obs_points.shape[0], 1, 1).to(self.device)

        observations = [obs_points, obs_labels, supp_points, supp_labels]
        query_points = torch.cat([grid_points, iou_points], dim = -2)

        with torch.no_grad():
            fx = self.model(observations, query_points, chunk_size = chunk_size).detach().cpu().squeeze(dim = -1)  

        fx_volume = fx[:, :grid_points.shape[1]].reshape(fx.shape[0], voxel_resolution, voxel_resolution, voxel_resolution)
        fx_iou = fx[:, grid_points.shape[1]:]

        mesh, metrics = compute_evaluation_metrics(fx_volume, fx_iou, chamfer_points, chamfer_normals, iou_occs, shape_ids, scales)
        
        return mesh, metrics