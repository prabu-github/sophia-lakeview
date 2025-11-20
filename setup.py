import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
from itertools import product
from pprint import pprint
# import argparse
import shutil 

hytraits_path = (Path(__file__).parent.parent/'hytraits').resolve()
if str(hytraits_path) not in sys.path:
    sys.path.append(str(hytraits_path))
import hytraits as H 
from utils import (get_setup_args,
                   get_paths,
                   make_compatible_csvs,
                   make_color_configs,
                   make_transform_configs,
                   make_metric_configs,
                   make_train_configs,
                   make_deploy_configs,
                   make_eda_configs,
                   cleanup)


if __name__ == '__main__':
    args = get_setup_args('sophia-lakeview: traitmodeling')
    verbose = args['verbose']
    
    PATHS = get_paths()
    csv_file = PATHS['original']/'LakeViewDF_v3.csv'
    transformations_file = PATHS['original']/'transform_v3.csv'
    
    # compatible CSV(s)
    (df, counts_df) = make_compatible_csvs(csv_file=csv_file, 
                                           verbose=verbose)

    # make color config(s)
    make_color_configs(df=df, verbose=verbose)

    # make transform config(s)
    make_transform_configs(verbose=verbose)

    # make metric config(s)
    make_metric_configs(verbose=verbose)

    # make train config(s)
    make_train_configs(transformations_file=transformations_file, 
                       n_inners=args['n_inners'],
                       n_outers=args['n_outers'],
                       test_percent=args['test_percent'],
                       valid_percent=args['valid_percent'],
                       verbose=verbose)

    # make ideploy config(s)
    if args['deploy']:
        make_deploy_configs(verbose=verbose)

    if args['eda']:
        make_eda_configs(transformations_file=transformations_file, 
                         verbose=verbose)
        
    if args['cleanup']:
        cleanup()
