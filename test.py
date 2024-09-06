import torch
import argparse

from utils.common import seed_everything
from utils.io import read_config
from utils.setup_model import make_trainer


def main(args, cfg):
    if torch.cuda.is_available():
        device = 0
    else:
        print('Evaluation requires a gpu.'); exit()

    seed_everything(cfg['misc']['manual_seed'])

    learnable = False if cfg['model']['kernel']['input_type'] == 'only_points' else True

    # Load checkpoint.
    if learnable:
        if not args.ckpt: print('Please provide a checkpoint for learnable kernels.'); exit()

        checkpoint = torch.load(args.ckpt, map_location = {'cuda:0': f'cuda:{device}'})
        print(f'Loaded checkpoint from {args.ckpt}.')
    else: checkpoint = None

    trainer = make_trainer(cfg)
    trainer.init_testing(device, checkpoint)

    trainer.test(print_metrics = args.print_metrics, save_reconstructions = args.save_reconstructions)
  

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('config', type = str, 
                           help = 'The path to the config file.')
    argparser.add_argument('--ckpt', type = str,
                           help = 'The path to the checkpoint file. Must only be provided for learnable kernels.', default = '')
    argparser.add_argument('--print_metrics', action = 'store_true',
                           help = 'Whether to print metrics during inference.')
    argparser.add_argument('--save_reconstructions', action = 'store_true',
                           help = 'Whether to save all reconstructed meshes along with input point clouds during inference.')

    args, _ = argparser.parse_known_args()
    cfg = read_config(args.config)
    
    main(args, cfg)