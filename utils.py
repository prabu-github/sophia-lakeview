from __future__ import annotations
from typing import List, Dict
from pathlib import Path
import argparse


def get_seed() -> int:
    '''
    Return: int
            A common seed.
    '''
    return 2147483647 


def get_paths() -> Dict:
    '''
    Return: Dict {key: Path}
            Project specific paths.
    '''
    io_dir = Path(__file__).parent/'io_'
    return {'io': io_dir,
            'original': io_dir/'data',
            'compatible': io_dir/'compatible',
            'model': io_dir/'model', 
            'deploy': io_dir/'deploy',
            'post': io_dir/'post',
            'eda': io_dir/'eda',
            'chtc': io_dir.parent/'chtc'}


def get_cli_args() -> Dict:
    '''
    CLI argument parser.

    Return: Dict
            Dictionary of CLI args.
    '''
    parser = argparse.ArgumentParser('Trait modeling.')
    parser.add_argument('--pattern', 
                        action='store', 
                        type=str, 
                        default='*')
    parser.add_argument('--min_n_samples', 
                        action='store', 
                        type=int, 
                        default=30)
    parser.add_argument('--n_outers', 
                        action='store', 
                        default=200, 
                        type=int)
    parser.add_argument('--n_inners', 
                        action='store', 
                        default=30, 
                        type=int)
    parser.add_argument('--outer_type', 
                        action='store', 
                        default='montecarlo')
    parser.add_argument('--inner_type', 
                        action='store', 
                        default='montecarlo')
    parser.add_argument('--test_percent', 
                        action='store', 
                        default=15, 
                        type=int)
    parser.add_argument('--valid_percent', 
                        action='store', 
                        default=15, 
                        type=int)
    parser.add_argument('--n_components', 
                        action='store', 
                        default=40, 
                        type=int)
    parser.add_argument('--make_compatible', 
                        action='store_true', 
                        default=False)
    
    ##
    parser.add_argument('--chtc_n_models_per_submit', 
                        action='store', 
                        type=int, 
                        default=-1)
    parser.add_argument('--chtc_user', 
                        action='store', 
                        type=str, 
                        default='pravindran')
    parser.add_argument('--chtc_project_name', 
                        action='store', 
                        type=str, 
                        default='sophia-lakeview')
    parser.add_argument('--chtc_models_list',
                        action='store',
                        type=str,
                        default='')
    args = parser.parse_args().__dict__
    
    args['n_repeats'] = (args['n_outers'], args['n_inners'])
    args['split_types'] = (args['outer_type'], args['inner_type'])
    args['split_percents'] = (args['test_percent'], args['valid_percent'])
    return args
