from __future__ import annotations
import sys
from pathlib import Path
from pprint import pprint
import argparse

hytraits_path = (Path(__file__).parent.parent/'hytraits').resolve()
if str(hytraits_path) not in sys.path:
    sys.path.append(str(hytraits_path))
import hytraits as H
from paths import get_paths


def get_eda_files(trait_name: str,
                  eda_prefix: str) -> List[Path]:
    return []

if __name__ == '__main__':
    parser = argparse.ArgumentParser('sophia-lakeview: traiteda')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--eda_config_idx', action='store', type=int)
    group.add_argument('--eda_config_json', action='store', type=str)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args().__dict__

    paths = get_paths()

    config_jsons = [f for f in paths['config'].glob('EDA__*.json')]
    config_jsons.sort()

    eda_config_files = []
    if args['eda_config_idx'] is not None:
        if (args['eda_config_idx'] > len(config_jsons)):
            raise Exception(f'Invalid {args["eda_config_idx"] = }.')
        if args['eda_config_idx'] < 0:
            eda_config_files = config_jsons
        else:
            eda_config_files.append(config_jsons[args['eda_config_idx']])

    if args['eda_config_json'] is not None:
        if not (paths['config']/args['eda_config_json'].exists()):
            raise Exception(f'Invalid {str(args["eda_config_json"]) = }.')
        eda_config_files.append(paths['config']/args['eda_config_json'])

    for eda_config_file in eda_config_files:
        print(f'====== EDA: {eda_config_file.name} ...')
        EDA = H.UnivarAnalysis()
        EDA(config_json=eda_config_file)
