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
    parser = argparse.ArgumentParser('sophia-lakeview: deploy')
    parser.add_argument('--config_idx', action='store', type=int, required=True)
    args = parser.parse_args().__dict__

    config_jsons = [f for f in Path('io/config').glob('DEPLOY*.json')]
    config_jsons.sort()
    if (args['config_idx'] < 0) or (args['config_idx'] > len(config_jsons)):
        raise Exception(f'Invalid {args["config_idx"] = }.')
        
    config_json = config_jsons[args["config_idx"]]
    print(f'Deploying: {config_json.name} ({args["config_idx"]})...')
    DEPLOYER = H.TraitDeployerCSV()
    DEPLOYER(param_json=config_json)
