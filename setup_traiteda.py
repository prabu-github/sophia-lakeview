import sys
from pathlib import Path
import numpy as np
import pandas as pd
from pandas import DataFrame
import json
from itertools import product
from pprint import pprint
import argparse
import shutil 

hytraits_path = (Path(__file__).parent.parent/'hytraits').resolve()
if str(hytraits_path) not in sys.path:
    sys.path.append(str(hytraits_path))
import hytraits as H 
from paths import get_paths

seed = 2147483647 


if __name__ == '__main__':
    parser = argparse.ArgumentParser('sophia-lakeview: setup_traiteda')
    parser.add_argument('--cleanup', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args().__dict__

    paths = get_paths()
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    ##### Compatible CSVs
    csv_file = paths['original']/'LakeViewDF_v2.csv'
    if args['verbose']:
        print(f'--- Loading {csv_file} ...')
    orig_df = pd.read_csv(csv_file)
    
    orig_wave_cols, comp_wave_cols = [], []
    for c in orig_df.columns:
        try:
            fc = float(c)
            orig_wave_cols.append(c)
            comp_wave_cols.append(f'X_{float(c):0.3f}')
        except:
            pass
    
    orig_to_comp = [('IC', 'ic'),
                    ('NPOC', 'npoc'),
                    ('Cl(-)', 'cl'), 
                    ('SO4(2-)', 'so4'), 
                    ('Silica', 'si'), 
                    ('NO2/NO3 ', 'no23'), # note the space
                    ('SRP', 'srp'),
                    ('NH4(+)', 'nh4'),
                    ('TN', 'tn'),
                    ('TP', 'tp'),
                    ('TSS', 'tss'),
                    ('CF_chl', 'cfchl'),
                    ('CF_PC', 'cfpc'),
                    ('PC:chl', 'pcchl')]
    for (o, c) in orig_to_comp:
        orig_cols = ['sampleID'] + [o] + orig_wave_cols 
        comp_cols = ['sample_id'] + ['y_true'] + comp_wave_cols
        comp_df = orig_df[orig_cols].dropna()
        comp_df.columns = comp_cols
        comp_df = comp_df[comp_df['y_true'] > 0] # no negative trait values!

        comp_csv = paths['compatible']/f'{c}.csv'
        comp_df.to_csv(comp_csv, index=None)
        if args['verbose']:
            print(f'------ Compatibalized: {comp_csv.stem} ({comp_df.shape}).')
    if args['verbose']:
        print(f'--- Created compatible CSVs. ({len(orig_to_comp)})')

    ##### Transforms
    transforms = {'400-800-asis-uv': [H.KeepWavelengths(keep_ranges=[(399.99, 800.01)]),
                                      H.UnitVectorize()],
                  '400-800-move-uv': [H.KeepWavelengths(keep_ranges=[(399.99, 800.01)]),
                                      H.CommonMinReflectance(),
                                      H.UnitVectorize()]}
    n_transform_configs = 0
    for (k, t) in transforms.items():
        H.save_transforms(transforms=t, transforms_file=paths['config']/f'TRANSFORM__{k}.json') 
        n_transform_configs += 1
    if args['verbose']:
        print(f'--- Created transforms ({n_transform_configs}).')

    ##### EDA configs
    comp_csvs = [f for f in paths['compatible'].glob('*.csv')]
    transform_jsons = sorted([f for f in paths['config'].glob('TRANSFORM__*.json')])
    
    n_eda_configs = 0
    for comp_csv in comp_csvs:
        for transform_json in transform_jsons: 
            transform_key = transform_json.stem.split('__')[1]
            eda_name = f'{transform_key}__{comp_csv.stem}'
            config = {'seed': seed,
                      'eda_types': ['uvraw', 'uvndi'],
                      'eda_name': eda_name,
                      'eda_dir': str(paths['eda']/eda_name),
                      'train_csv': str(comp_csv),
                      'subsample': -1,
                      'reduce': 'none',
                      'metrics': ['r2', 'pearson_correlation'],
                      'transform_json': str(transform_json)}
            with open(paths['config']/f'EDA__{eda_name}.json', 'w') as writer:
                json.dump(config, writer)
            n_eda_configs += 1
    if args['verbose']:
        print(f'--- Created eda configs ({n_eda_configs}).')
    # clean up for debugging - run before packaging for CHTC.
    if args['cleanup']:
        for d in paths.values():
            if d.stem != 'original':
                shutil.rmtree(str(d))
