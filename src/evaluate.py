import os
from typing import List, Tuple, Dict

import hydra
import torch
import logging
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from models.ames import AMES
from utils.metrics import mean_average_precision_revisited_rerank
from utils.utils import set_seed
from utils.dataset_loader import get_test_loaders
import argparse
from omegaconf import OmegaConf
import sys

log = logging.getLogger(__name__)

def evaluate(
        model: nn.Module,
        query_loader: DataLoader,
        gallery_loader: DataLoader,
        lamb: List[int],
        temp: List[int],
        num_rerank: List[int]) -> Tuple[Dict[str, float], float]:
    model.eval()

    with torch.no_grad():
        torch.cuda.empty_cache()
        metrics = mean_average_precision_revisited_rerank(
            model,
            query_loader, gallery_loader, query_loader.dataset.cache_nn,
            lamb=lamb,
            temp=temp,
            top_k=num_rerank,
            gnd=query_loader.dataset.gnd_data,
        )
    return metrics


@hydra.main(config_path="../conf", config_name="test", version_base=None)
def main(cfg: DictConfig):
    device = torch.device('cuda:0' if torch.cuda.is_available() and not cfg.cpu else 'cpu')

    set_seed(cfg.seed)

    query_loader, gallery_loader = get_test_loaders(cfg.desc_name, cfg.test_dataset, cfg.num_workers)

    model = AMES(desc_name=cfg.desc_name, local_dim=cfg.dim_local_features, pretrained=cfg.model_path if not os.path.exists(cfg.model_path) else None, **cfg.model)

    if os.path.exists(cfg.model_path):
        checkpoint = torch.load(cfg.model_path, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint['state'], strict=True)

    model.to(device)
    model.eval()

    map = evaluate(model=model, lamb=cfg.test_dataset.lamb, temp=cfg.test_dataset.temp, num_rerank=cfg.test_dataset.num_rerank,
                   query_loader=query_loader, gallery_loader=gallery_loader)
    return map

def load_config_from_json(json_path):
    """Load configuration from JSON file and convert to OmegaConf"""
    import json
    from omegaconf import OmegaConf
    with open(json_path, 'r') as f:
        json_config = json.load(f)
    env_vars = json_config.get('configurations', [{}])[0].get('env', {})
    cfg = OmegaConf.create({
        'desc_name': env_vars.get('DESC_NAME', 'dinov2'),
        'test_dataset': {
            'name': env_vars.get('DATASET_NAME', 'roxford5k'),
            'desc_dir': env_vars.get('DESC_DIR', 'data/roxford5k'),
            'test_gnd_file': env_vars.get('TEST_GND_FILE', 'gnd_roxford5k.pkl'),
            'nn_file': env_vars.get('NN_FILE', 'nn_dinov2.pkl'),
            'db_desc_num': int(env_vars.get('DB_DESC_NUM', 700)),
            'query_desc_num': int(env_vars.get('QUERY_DESC_NUM', 700)),
            'batch_size': int(env_vars.get('BATCH_SIZE', 1)),
            'pin_memory': env_vars.get('PIN_MEMORY', 'true').lower() == 'true',
            'lamb': [float(x) for x in env_vars.get('LAMB', '0.0,1.0,2.0').split(',')],
            'temp': [float(x) for x in env_vars.get('TEMP', '0.5,1.0,2.0').split(',')],
            'num_rerank': [int(x) for x in env_vars.get('NUM_RERANK', '100,200,500').split(',')]
        },
        'num_workers': int(env_vars.get('NUM_WORKERS', 0)),
        'cpu': env_vars.get('CPU', 'false').lower() == 'true',
        'seed': int(env_vars.get('SEED', 42)),
        'dim_local_features': int(env_vars.get('DIM_LOCAL_FEATURES', 768)),
        'model_path': env_vars.get('MODEL_PATH', 'dinov2_ames.pt'),
        'model': {}
    })
    return cfg
if __name__ == '__main__':
    sys.argv = [
        'evaluate.py',
        'descriptors=dinov2',
        'data_root=data', 
        'model_path=dinov2_ames.pt',
        'num_workers=0'
    ]
    main()
    print(f"Evaluation result: {result}")
	'''
	# Check for JSON config file option
    launch_config_path = r"C:\github\ames\ames\.vscode\launch_workers0.json"
    use_json_config = os.path.exists(launch_config_path)
    
    if use_json_config:
        print(f"Loading configuration from: {launch_config_path}")
        cfg = load_config_from_json(launch_config_path)
        
        # Run main with JSON config
        result = main(cfg)
        print(f"mAP result: {result}")
        
    else:
        # For debugging - manual config
        if True:  # Set to True for debugging
            from omegaconf import OmegaConf
            
            # Create manual config for debugging
            cfg = OmegaConf.create({
                'desc_name': 'dinov2',
                'test_dataset': {
                    'name': 'roxford5k',
                    'desc_dir': 'data/roxford5k',
                    'test_gnd_file': 'gnd_roxford5k.pkl',
                    'nn_file': 'nn_dinov2.pkl',
                    'db_desc_num': 700,
                    'query_desc_num': 700,
                    'batch_size': 1,
                    'pin_memory': True,
                    'lamb': [0.0, 1.0, 2.0],
                    'temp': [0.5, 1.0, 2.0],
                    'num_rerank': [100, 200, 500]
                },
                'num_workers': 0,
                'cpu': False,
                'seed': 42,
                'dim_local_features': 768,
                'model_path': 'dinov2_ames.pt',
                'model': {}
            })
            
            # Run main with manual config
            result = main(cfg)
            print(f"mAP result: {result}")
        else:
            # Normal hydra execution
            main()  
	'''
