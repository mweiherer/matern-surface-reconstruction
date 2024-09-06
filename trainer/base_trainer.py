from abc import ABC
import torch
import os
import wandb
from tqdm import tqdm
import numpy as np
import datetime
from pycg import vis, render
import point_cloud_utils as pcu

from utils.common import dict_mean
from utils.io import cond_mkdir


class BaseTrainer(ABC):
    '''
    Abstract base class that takes care of training a given model. Every new trainer class should inherit
    from this class and override the following methods: train_step(), compute_loss(), and eval_step(). 
    You don't have to touch the rest!
    :param model: The model that should be trained
    :param optimizer: The optimizer used to train the model
    :param train_loader: The dataloader holding the training data
    :param val_loader: The dataloader holding the validation data
    :param test_loader: The dataloader holding the test data
    :param cfg: The config file
    '''
    def __init__(self, model, optimizer, train_loader, val_loader, test_loader, cfg):
        super().__init__()

        self.model = model
        self.optimizer = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.cfg = cfg

    def init_training(self, device, checkpoint):
        '''
        Initializes the training (optionally from a given checkpoint).
        :param device: The device it should be trained on
        :param checkpoint: Optional checkpoint from which training should start
        '''
        self.device = device
        self.model = self.model.to(self.device)

        self.checkpoint_dir = f'./wandb/nkf-shapenet/{wandb.run.id}'
        cond_mkdir(self.checkpoint_dir)

        self.epochs_run = 0
        if checkpoint is not None:
            self._load_checkpoint(checkpoint)

        # Cache training metrics per epoch and evaluation metrics for wandb logging.
        self.train_metrics = None
        self.eval_metrics = None

    def init_testing(self, device, checkpoint):
        '''
        Initializes testing (optionally from a given checkpoint).
        :param device: The device it should be tested on
        :param checkpoint: Optional checkpoint which should be used for testing
        '''
        self.device = device
        self.model = self.model.to(self.device)

        if checkpoint is not None:
            self._load_checkpoint(checkpoint)

    def _new_epoch(self):
        '''
        Initializes a new epoch.
        '''
        self.train_metrics = {}
        self.eval_metrics = {}
    
    def train_step(self, batch_data):
        '''
        Performs a single training step on the given batch data. This method is model-specific
        and should be adapted accordingly. 
        :param batch_data: A batch of training data
        '''
        raise NotImplementedError
    
    def compute_loss(self, batch_data):
        '''
        Computes the loss for the given batch data.
        Should return the individual loss terms as dictionary.
        :param batch_data: A batch of training data
        '''
        raise NotImplementedError
    
    def eval_step(self, batch_data):
        '''
        Performs a single evaluation step on the given batch data. This method is model-specific
        and should be adapted accordingly. 
        :param batch_data: A batch of evaluation data
        '''
        raise NotImplementedError
    
    def train(self, num_epochs):
        '''
        Training loop.
        :param num_epochs: The number of epochs to train
        '''
        self.model.train(); 

        for epoch in range(self.epochs_run, self.epochs_run + num_epochs):
            self._new_epoch()

            for batch_data in tqdm(self.train_loader):
                try:
                    self.train_step(batch_data)
                except Exception as exception:
                    print(exception); continue
                
            if epoch % self.epochs_til_evaluation == 0:
                self.evaluate(); self.model.train()

            self.log_epoch(epoch)

    def evaluate(self):
        '''
        Performs model evaluation.
        '''
        self.model.eval(); print('Perform model evaluation.')

        # Randomly select one batch (= mesh) for visualization.
        rand_idx = torch.randint(len(self.val_loader), (1,))

        for idx, batch_data in enumerate(tqdm(self.val_loader)):
            mesh, metrics = self.eval_step(batch_data)
 
            if idx == rand_idx:
                # Render mesh and log it to wandb.
                mesh = vis.mesh(mesh[0][1]['vertices'], mesh[0][1]['faces'])
                rendering = render.multiview_image([mesh])
                self.to_eval_log({'metrics_dict': metrics, 'rendering': rendering})
            else:
                self.to_eval_log({'metrics_dict': metrics})
        
    def test(self, print_metrics, save_reconstructions):
        '''
        Performs model inference/testing.
        :param print_metrics: If true, print metrics 
        :param save_reconstructions: If true, save all reconstructed meshes and their inputs
        '''
        self.model.eval()

        current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        base_path_recons = f'./test-{current_timestamp}/reconstructions'
        base_path_inputs = f'./test-{current_timestamp}/inputs'
        cond_mkdir(base_path_recons); cond_mkdir(base_path_inputs)

        metrics = []
        for batch_data in tqdm(self.test_loader):
            batch_mesh, batch_metrics = self.eval_step(batch_data)
            if print_metrics: print(batch_data['shape_id'][0], [f'{k}: {v.item()}' for k, v in batch_metrics.items()])
          
            if save_reconstructions: 
                f_name_recon = os.path.join(base_path_recons, f"{batch_data['shape_id'][0]}.ply") 
                f_name_input = os.path.join(base_path_inputs, f"{batch_data['shape_id'][0]}.ply") 

                pcu.save_mesh_vfn(f_name_recon, batch_mesh[0][1]['vertices'], batch_mesh[0][1]['faces'], batch_mesh[0][1]['normals'])
                pcu.save_mesh_v(f_name_input, batch_data['obs_points'][0].numpy())
    
            metrics.append(batch_metrics)
        
        mean = dict_mean(metrics)

        print('===== Test metrics =====')
        for k, v in mean.items():
            print(f'{k} ({len(metrics)}): {v.item()}')

    def to_train_log(self, log_dict):
        '''
        Saves per-batch logging data to train_metrics.
        :param log_dict: Dictionary containing logging data as dictionaries or PyTorch tensors
        '''
        if self.train_metrics is None:
            raise Exception('Can not log data before new_epoch() has been called first.')

        for key, value in log_dict.items():
            if not (isinstance(value, dict) or isinstance(value, torch.Tensor)):
                raise ValueError(f'Can not log data of type {type(value)}. Only dict and torch.Tensor is allowed.')

            if key not in self.train_metrics:
                self.train_metrics[key] = []

            self.train_metrics[key].append(value)

    def to_eval_log(self, log_dict):
        '''
        Saves per-batch logging data to eval_metrics.
        :param log_dict: Dictionary containing logging data as dictionaries 
                         or an image as numpy array
        '''
        if self.eval_metrics is None:
            raise Exception('Can not log data before new_epoch() has been called first.')
        
        for key, value in log_dict.items():
            if not (isinstance(value, dict) or isinstance(value, np.ndarray)):
                raise ValueError(f'Can not log data of type {type(value)}. Only dict or image as numpy array is allowed.')
            
            if isinstance(value, dict):
                value = {f'val/{key}': value for key, value in value.items()}
            
            if key not in self.eval_metrics:
                self.eval_metrics[key] = []

            self.eval_metrics[key].append(value)

    def log_epoch(self, epoch, additional_logs = None):
        '''
        Computes per-epoch metrics from the collected per-batch train and eval data 
        and writes collected data to wandb. Also saves checkpoint to disk.
        :param epoch: The epoch
        :param additional_logs: Optional, per-epoch logs as dict that should be uploaded to wandb
        '''
        wandb_metrics = {}

        # First, we collect training metrics.
        for key, value in self.train_metrics.items():
            if all(isinstance(x, dict) for x in value):
                wandb_metrics.update(dict_mean(value))
            else: # If it's not a list of dicts, must be list of PyTorch tensors. See to_train_log().
                wandb_metrics[key] = torch.Tensor(value).mean()
    
        if additional_logs is not None:
            wandb_metrics.update(additional_logs)

        # Next, collect evaluation metrics.
        for key, value in self.eval_metrics.items():
            if all(isinstance(x, dict) for x in value):
                wandb_metrics.update(dict_mean(value))
            elif all(isinstance(x, torch.Tensor) for x in value):
                wandb_metrics[key] = torch.Tensor(value).mean()
            else: # If it's not a list of dicts, must be an image as numpy array. See to_eval_log().
                wandb_metrics[key] = [wandb.Image(x) for x in value]

        eval_iou = wandb_metrics.get('val/iou', None) 
        self._save_checkpoint(epoch, eval_iou)

        wandb.log(wandb_metrics, step = epoch)

        self.train_metrics = None
        self.eval_metrics = None

    def _save_checkpoint(self, epoch, eval_iou):
        '''
        Saves model checkpoints and optimizer state dict to disk.
        :param epoch: The epoch
        :param eval_iou: The IoU score on the validation set
        '''
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'last.pth'))
        
        if eval_iou is not None:
            torch.save(checkpoint, 
                       os.path.join(self.checkpoint_dir,
                                    f'epoch={str(epoch).zfill(2)}-iou={str(float(eval_iou))[:5]}.pth'))

    def _load_checkpoint(self, checkpoint):
        '''
        Loads model checkpoints and optimizer state dict.
        :param checkpoint: The checkpoint
        '''  
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epochs_run = checkpoint['epoch']