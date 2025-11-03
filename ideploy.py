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

np.set_printoptions(precision=8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('sophia-lakeview: ideploy')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config_idx', action='store', type=int)
    group.add_argument('--config_json', action='store', type=str)
    args = parser.parse_args().__dict__

    ideploy_configs_dir = Path(__file__).parent/'io/config/ideploy'
    config_jsons = [f for f in ideploy_configs_dir.glob('IDEPLOY*.json')]
    config_jsons.sort()

    if args['config_idx'] is not None:
        if (args['config_idx'] < 0) or (args['config_idx'] > len(config_jsons)):
            raise Exception(f'Invalid {args["config_idx"] = }.')
        config_json = config_jsons[args['config_idx']]
        
    if args['config_json'] is not None:
        if not (ideploy_configs_dir/args['config_json']).exists():
            raise Exception(f'Invalid {str(args["config_json"]) = }.')
        config_json = ideploy_configs_dir/args['config_json'] 

    if args['config_idx'] is not None:
        print(f'iDeploying: {config_json.name} ({args["config_idx"]}) ...')
    else:
        print(f'iDeploying: {config_json.name} ...')
        
    DEPLOYER = H.TraitDeployerCSV()
    DEPLOYER(param_json=config_json)
