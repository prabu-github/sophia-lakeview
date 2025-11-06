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
    print('Setting up ...')
    parser = argparse.ArgumentParser('sophia-lakeview: setup')
    parser.add_argument('--model_type', action='store', type=str, default='plsr')
    parser.add_argument('--model_select', action='store', type=str, default='median-min')
    parser.add_argument('--n_components', action='store', type=int, default=30)
    parser.add_argument('--n_outers', action='store', type=int, default=200)
    parser.add_argument('--n_inners', action='store', type=int, default=50)
    parser.add_argument('--test_percent', action='store', type=int, default=15)
    parser.add_argument('--valid_percent', action='store', type=int, default=15)
    parser.add_argument('--ideploy', action='store_true', default=False)
    parser.add_argument('--edeploy', action='store_true', default=False)
    parser.add_argument('--cleanup', action='store_true', default=False)
    args = parser.parse_args().__dict__

    paths = get_paths()
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    ##### Compatible CSVs
    csv_file = paths['original']/'LakeViewDF_v2.csv'
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
        print(f'------ Compatibalized: {comp_csv.stem} ({comp_df.shape}).')
    print(f'--- Created compatible CSVs. ({len(orig_to_comp)})')
    
    ##### Transforms
    transforms = {'400-800-asis-uv': [H.KeepWavelengths(keep_ranges=[(399.99, 800.01)]),
                                      H.UnitVectorize()],
                  '410-800-asis-uv': [H.KeepWavelengths(keep_ranges=[(409.99, 800.01)]),
                                      H.UnitVectorize()],
                  '420-800-asis-uv': [H.KeepWavelengths(keep_ranges=[(419.99, 800.01)]),
                                      H.UnitVectorize()],
                  '430-800-asis-uv': [H.KeepWavelengths(keep_ranges=[(429.99, 800.01)]),
                                      H.UnitVectorize()],
                  '440-800-asis-uv': [H.KeepWavelengths(keep_ranges=[(439.99, 800.01)]),
                                      H.UnitVectorize()], 
                  '450-800-asis-uv': [H.KeepWavelengths(keep_ranges=[(449.99, 800.01)]),
                                      H.UnitVectorize()],
                 
                  '400-800-move-uv': [H.KeepWavelengths(keep_ranges=[(399.99, 800.01)]),
                                      H.Offset(),
                                      H.UnitVectorize()],
                  '410-800-move-uv': [H.KeepWavelengths(keep_ranges=[(409.99, 800.01)]),
                                      H.Offset(),
                                      H.UnitVectorize()],
                  '420-800-move-uv': [H.KeepWavelengths(keep_ranges=[(419.99, 800.01)]),
                                      H.Offset(),
                                      H.UnitVectorize()],
                  '430-800-move-uv': [H.KeepWavelengths(keep_ranges=[(429.99, 800.01)]),
                                      H.Offset(),
                                      H.UnitVectorize()],
                  '440-800-move-uv': [H.KeepWavelengths(keep_ranges=[(439.99, 800.01)]),
                                      H.Offset(),
                                      H.UnitVectorize()], 
                  '450-800-move-uv': [H.KeepWavelengths(keep_ranges=[(449.99, 800.01)]),
                                      H.Offset(),
                                      H.UnitVectorize()],}
    n_transform_configs = 0
    for (k, t) in transforms.items():
        H.save_transforms(transforms=t, transforms_file=paths['config']/f'TRANSFORM__{k}.json') 
        n_transform_configs += 1
    print(f'--- Created transforms ({n_transform_configs}).')
    
    ##### Metrics
    metrics = [H.R2(),
               H.RangeNormalizedRMSE(),
               H.InterquartileNormalizedRMSE(),
               H.RMSE()]
    H.save_metrics(metrics=metrics, metrics_file=paths['config']/'METRIC__r2-rnrmse-inrmse-rmse.json') 
    print('--- Created metrics.')
    
    
    ##### Colors
    sids = sorted(list(set(orig_df['sampleID'].values.tolist())))
    colors = ['#CC99CC']*len(sids)
    colors_dict = {s: c for (s, c) in zip(sids, colors)}
    with open(paths['config']/'COLOR__default.json', 'w') as writer:
        json.dump(colors_dict, writer, indent=4)
    print('--- Created colors.')
    
    
    ##### Train config
    train_csvs = sorted([f for f in paths['compatible'].glob('*.csv')])
    transform_jsons = sorted([f for f in paths['config'].glob('TRANSFORM__*.json')])

    n_train_configs = 0
    for (train_csv, transform_json) in product(train_csvs, transform_jsons):
        transform_key = transform_json.stem.split('__')[1]
        model_name = f'{args["model_type"]}__{args["model_select"]}__{transform_key}__{train_csv.stem}-avg'
        train_config = {'seed': seed,
                        'model_name': model_name,
                        'model_type': args['model_type'],
                        'model_dir': str(paths['model']/model_name),
                        'model_selection': args['model_select'],
                        'n_components': args['n_components'],
                        
                        'n_outers': args['n_outers'],
                        'n_inners': args['n_inners'],
                        'outer_type': 'montecarlo',
                        'inner_type': 'montecarlo',
                        'test_percent': args['test_percent'],
                        'valid_percent': args['valid_percent'], 
                                  
                        'train_csv': str(train_csv),
                        'subsample': -1,
                        'reduce': 'mean',
                        'transform_json': str(transform_json)}
        
        config_json = paths['config']/f'TRAIN__{model_name}.json'
        with open(config_json, 'w') as writer:
            json.dump(train_config, writer)
        n_train_configs += 1
    print(f'--- Created train configs ({n_train_configs}).')
    
    ##### Internal deploy config
    if args['ideploy']:
        n_deploy_configs = 0
        for (train_csv, transform_json) in product(train_csvs, transform_jsons):
            transform_key = transform_json.stem.split('__')[1]
            model_name = f'{args["model_type"]}__{args["model_select"]}__{transform_key}__{train_csv.stem}-avg'
            with open(paths['config']/f'TRAIN__{model_name}.json', 'r') as reader:
                train_config = json.load(reader) 
        
            # internal eval
            deploy_csv = Path(train_config['train_csv'])
            deploy_other_cols = []
            deploy_split_label = 'TEST'
            deploy_name = f'{model_name}__{deploy_split_label}-{deploy_csv.stem}'
            deploy_config = {'deploy_dir': str(paths['ideploy']/deploy_name),
                             'deploy_csv': str(deploy_csv),
                             'deploy_other_cols': deploy_other_cols,
                             'deploy_subsample': -1,
                             'deploy_reduce': 'mean',
                             'deploy_split_label': deploy_split_label, 
                             'color_json': str(paths['config']/'COLOR__default.json'),
                             'metric_json': str(paths['config']/'METRIC__r2-rnrmse-inrmse-rmse.json')} 
            deploy_config.update(train_config)
            config_json = paths['config']/f'IDEPLOY__{deploy_name}.json'
            with open(config_json, 'w') as writer:
                json.dump(deploy_config, writer)
            n_deploy_configs += 1
        print(f'--- Created internal deploy configs ({n_deploy_configs}).')

    ##### External deploy config
    if args['edeploy']:
        # this is only for testing purposes for now!!
        n_deploy_configs = 0
        for (train_csv, transform_json) in product(train_csvs, transform_jsons):
            transform_key = transform_json.stem.split('__')[1]
            model_name = f'{args["model_type"]}__{args["model_select"]}__{transform_key}__{train_csv.stem}-avg'
            with open(paths['config']/f'TRAIN__{model_name}.json', 'r') as reader:
                train_config = json.load(reader) 
        
            # internal eval
            deploy_csv = Path(train_config['train_csv'])
            deploy_other_cols = []
            deploy_split_label = 'TEST'
            deploy_name = f'{model_name}__{deploy_split_label}-{deploy_csv.stem}'
            deploy_config = {'deploy_dir': str(paths['edeploy']/deploy_name),
                             'deploy_csv': str(deploy_csv),
                             'deploy_other_cols': deploy_other_cols,
                             'deploy_subsample': -1,
                             'deploy_reduce': 'mean',
                             'deploy_split_label': deploy_split_label, 
                             'color_json': str(paths['config']/'COLOR__default.json'),
                             'metric_json': str(paths['config']/'METRIC__r2-rnrmse-inrmse-rmse.json')} 
            deploy_config.update(train_config)
            config_json = paths['config']/f'EDEPLOY__{deploy_name}.json'
            with open(config_json, 'w') as writer:
                json.dump(deploy_config, writer)
            n_deploy_configs += 1
        print(f'--- Created external deploy configs ({n_deploy_configs}).')

    # clean up for debugging - run before packaging for CHTC.
    if args['cleanup']:
        for d in paths.values():
            if d.stem != 'original':
                shutil.rmtree(str(d))
