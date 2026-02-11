from src.models.mf_elvis import MF_ELVis
from src.models.elvis import ELVis
from src.models.presley import PRESLEY
from src.models.collei import COLLEI

from src.datamodule import *


def get_model(model_name, config, nusers):
    if model_name == 'MF_ELVis':
        model = MF_ELVis(d=config['d'],
                         nusers=nusers,
                         lr=config['lr'])
    elif model_name == 'ELVis':
        model = ELVis(d=config['d'],
                      nusers=nusers,
                      lr=config['lr'])
    elif model_name == 'PRESLEY':
        model = PRESLEY(d=config['d'],
                       nusers=nusers,
                       lr=config['lr'],
                        dropout=config['dropout'],
                        debug_sanity=config.get('debug_sanity', False),
                        debug_sanity_freq=config.get('debug_sanity_freq', 200),
                        ds_no_rho=config.get('ds_no_rho', False),
                        debug_overfit=config.get('debug_overfit', False))
    elif model_name == 'COLLEI':
        model = COLLEI(d=config['d'],
                       nusers=nusers,
                       lr=config['lr'],
                       tau=config['tau'])
    return model


def get_presley_config(config, nusers):
    return PRESLEY(config=config, nusers=nusers)


def get_dataset_constructor(model_name):
    if model_name in ['MF_ELVis', 'ELVis']:
        dataset = TripadvisorImageAuthorshipBCEDataset
    elif model_name in ['COLLEI']:
        dataset = TripadvisorImageAuthorshipCLDataset
    elif model_name in ['PRESLEY']:
        dataset = TripadvisorImageAuthorshipBPRDataset
    return dataset


import torch

def count_trainable_params(model):
    """
    Count number of trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_model_size_mb(model, bytes_per_param=4):
    """
    Estimate model size in MB assuming float32 parameters by default.
    """
    n_params = count_trainable_params(model)
    return n_params * bytes_per_param / (1024 ** 2)
