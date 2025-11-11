from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from pandas import DataFrame
from pprint import pprint
import time
import argparse

hytraits_path = (Path(__file__).parent.parent/'hytraits').resolve()
if str(hytraits_path) not in sys.path:
    sys.path.append(str(hytraits_path))
import hytraits as H
from paths import get_paths


def get_deploy_files(train_config_file: Path,
                     deploy_prefix: str) -> List[Path]:
    model_name = '__'.join(train_config_file.stem.split('__')[1:])
    patt = f'{deploy_prefix}__{model_name}__*.json'
    return [f for f in train_config_file.parent.glob(patt)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('sophia-lakeview: traitmodeling')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train_config_idx', action='store', type=int)
    group.add_argument('--train_config_json', action='store', type=str)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--ideploy', action='store_true', default=False)
    parser.add_argument('--edeploy', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args().__dict__

    paths = get_paths()
    train_config_jsons = [f for f in paths['config'].glob('TRAIN__*.json')]
    train_config_jsons.sort()

    if args['train_config_idx'] is not None:
        if (args['train_config_idx'] < 0) or (args['train_config_idx'] > len(train_config_jsons)):
            raise Exception(f'Invalid {args["train_config_idx"] = }.')
        train_config_file = train_config_jsons[args['train_config_idx']]
        
    if args['train_config_json'] is not None:
        if not (paths['config']/args['train_config_json']).exists():
            raise Exception(f'Invalid {str(args["config_json"]) = }.')
        train_config_file = paths['config']/args['train_config_json'] 


    # Training
    if args['train']:
        if args['train_config_idx'] is not None:
            print(f'====== Training: {train_config_file.name} ({args["train_config_idx"]}) ...')
        else:
            print(f'====== Training: {train_config_file.name} ...')
        
        TRAINER = H.TraitTrainer()
        TRAINER(config_json=train_config_file)
        
    # Internal deployment
    if args['ideploy']:
        deploy_config_files = get_deploy_files(train_config_file=train_config_file, 
                                               deploy_prefix='IDEPLOY')
        for deploy_config_file in deploy_config_files:
            print(f'====== iDeploying: {deploy_config_file.name} ...')
            DEPLOYER = H.TraitDeployerCSV()
            DEPLOYER(config_json=deploy_config_file)
            
    # External deployment:
    if args['edeploy']:
        deploy_config_files = get_deploy_files(train_config_file=train_config_file, 
                                               deploy_prefix='EDEPLOY')
        for deploy_config_file in deploy_config_files:
            print(f'====== eDeploying: {deploy_config_file.name} ...')
            DEPLOYER = H.TraitDeployerCSV()
            DEPLOYER(config_json=deploy_config_file)
