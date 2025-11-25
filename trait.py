from __future__ import annotations
import sys
from pathlib import Path
from pprint import pprint

hytraits_path = (Path(__file__).parent.parent/'hytraits').resolve()
if str(hytraits_path) not in sys.path:
    sys.path.append(str(hytraits_path))
import hytraits as H
from utils import (get_trait_args,
                   get_paths,
                   get_train_config_json,
                   get_deploy_config_jsons,
                   get_eda_config_json)


if __name__ == '__main__':
    args = get_trait_args()
    verbose = args['verbose']

    PATHS = get_paths()

    train_json = get_train_config_json(train_config_idx=args['train_config_idx'],
                                       train_config_stem=args['train_config_stem'])
    if train_json is not None:
        message = f'====== Training: {train_json.name}'
        if args['train_config_idx'] is not None:
            message += f' (idx = {args["train_config_idx"]}) ...'
        print(message)
        
        TRAINER = H.TraitTrainer()
        TRAINER(config_json=train_json)

    deploy_jsons = get_deploy_config_jsons(train_config_idx=args['train_config_idx'],
                                           train_config_stem=args['train_config_stem'])
    if deploy_jsons is not None:
        message = f'====== Deploying: {train_json.name}'
        if args['train_config_idx'] is not None:
            message += f' (idx = {args["train_config_idx"]}) ...'
        message += f' ({len(deploy_jsons)})'
        print(message)

        for deploy_json in deploy_jsons:
            DEPLOYER = H.TraitDeployerCSV()
            DEPLOYER(config_json=deploy_json)

    eda_json = get_eda_config_json(eda_config_idx=args['eda_config_idx'],
                                   eda_config_stem=args['eda_config_stem'])
    if eda_json is not None:
        message = f'====== EDA: {eda_json.name}'
        if args['eda_config_idx'] is not None:
            message += f' (idx = {args["eda_config_idx"]}) ...'
        print(message)
        
        EDA = H.EDA()
        EDA(config_json=eda_json)
