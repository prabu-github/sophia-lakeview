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
    parser = argparse.ArgumentParser('sophia-lakeview: train')
    parser.add_argument('--config_idx', action='store', type=int, required=True)
    args = parser.parse_args().__dict__

    config_dir = Path(__file__).parent/'io/config'
    config_jsons = [f for f in config_dir.glob('TRAIN*.json')]
    config_jsons.sort()
    print(f'{len(config_jsons) = }')
    if (args['config_idx'] < 0) or (args['config_idx'] > len(config_jsons)):
        raise Exception(f'Invalid {args["config_idx"] = }.')

    print(f'{args["config_idx"] = }')
    config_json = config_jsons[args["config_idx"]]
    print(f'Training: {config_json.name} ({args["config_idx"]})...')
    TRAINER = H.TraitTrainer()
    TRAINER(param_json=config_json) 
