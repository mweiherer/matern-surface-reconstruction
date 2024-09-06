import torch
import torchvision

from datasets import ShapeNetDataset
from datasets.transforms import ShapeReconstruction

from models.encoder import GridEncoder
from models.decoder import GridDecoder

from models import NeuralKernelField
from models.feature_field import FeatureField
from models.kernels import NeuralSplinesKernel, MaternKernel
from models.weight_models import SimpleWeightModel

from trainer import NeuralKernelFieldTrainer

from utils.common import seed_everything


dataset_dict = {
    'shapenet': ShapeNetDataset
}

kernel_dict = {
    'neural_splines': NeuralSplinesKernel,
    'matern': MaternKernel
}

module_dict = {
    'nkf': (NeuralKernelField, NeuralKernelFieldTrainer)
}

weight_model_dict = {
    'simple': SimpleWeightModel
}

encoder_dict = {
    'grid_encoder': GridEncoder
}

decoder_dict = {
    'grid_decoder': GridDecoder
}


def make_dataset(cfg):
    dataset = cfg['data']['dataset']

    dataset_transform = torchvision.transforms.Compose([ShapeReconstruction(cfg)])

    DatasetClass = dataset_dict[dataset]

    train_dataset = DatasetClass(cfg, mode = 'train', transform = dataset_transform)
    val_dataset = DatasetClass(cfg, mode = 'val', transform = dataset_transform)  
    test_dataset = DatasetClass(cfg, mode = 'test', transform = dataset_transform)  

    return train_dataset, val_dataset, test_dataset

def make_dataloaders(cfg):
    train_dataset, val_dataset, test_dataset = make_dataset(cfg)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = cfg['train']['batch_size'],
                                               shuffle = True,
                                               num_workers = cfg['train']['num_workers'],
                                               pin_memory = True,
                                               worker_init_fn = 
                                                    lambda w_id: seed_everything(cfg['misc']['manual_seed'] + w_id))
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size = 1,
                                             shuffle = False,
                                             pin_memory = True,
                                             num_workers = 1,
                                             worker_init_fn = 
                                                    lambda w_id: seed_everything(cfg['misc']['manual_seed'] + w_id))
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size = 1,
                                              shuffle = False,
                                              pin_memory = True,
                                              num_workers = 1,
                                              worker_init_fn = 
                                                    lambda w_id: seed_everything(cfg['misc']['manual_seed'] + w_id))

    return train_loader, val_loader, test_loader

def make_model(cfg, checkpoint = None):
    # Set output_dim of encoder to num_features.
    cfg['model']['feature_field']['encoder']['output_dim'] = cfg['model']['feature_field']['num_features']

    encoder = encoder_dict[cfg['model']['feature_field']['encoder']['type']](cfg)
    decoder = decoder_dict[cfg['model']['feature_field']['decoder']['type']](cfg)
    feature_field = FeatureField(encoder, decoder)
    
    kernel = kernel_dict[cfg['model']['kernel']['type']](cfg)
    weight_model_type = cfg['model']['weight_model']['type']

    if weight_model_type == 'none':
        weight_model = None
    else:
        weight_model = weight_model_dict[weight_model_type](cfg)

    model = module_dict[cfg['module']][0](feature_field, kernel, weight_model, cfg)

    # The whole thing runs only when double precision is used.
    model = model.double() 

    if checkpoint is not None:
        model.load_state_dict(checkpoint)

    return model

def make_optimizer(cfg, model):
    method = cfg['optimizer']['method']

    if method == 'adam':
        return torch.optim.Adam(model.parameters(),
                                lr = cfg['optimizer']['lr'],
                                betas = cfg['optimizer']['optimizer_kwargs']['betas'])
    if method == 'sgd':
        return torch.optim.SGD(model.parameters(), 
                               lr =  cfg['optimizer']['lr'],
                               momentum = cfg['optimizer']['optimizer_kwargs']['momentum'])
    
    raise ValueError(f"'{method}' is not a supported optimizer at the moment. Choose either 'adam' or 'sgd'.")

def make_trainer(cfg):
    '''
    Setup training from given config file.
    :param cfg: The config file
    :return trainer: The trainer that is responsible for training the given model
    '''
    model = make_model(cfg) 
    optimizer = make_optimizer(cfg, model)
    train_loader, val_loader, test_loader = make_dataloaders(cfg)

    trainer = module_dict[cfg['module']][1](model, optimizer, train_loader, val_loader, test_loader, cfg)

    return trainer