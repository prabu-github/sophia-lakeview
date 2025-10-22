import sys
from pathlib import Path
import numpy as np
import pandas as pd
from pandas import DataFrame
import json
from itertools import product
from pprint import pprint

hytraits_path = Path('../hytraits').resolve()
if str(hytraits_path) not in sys.path:
    sys.path.append(str(hytraits_path))
import hytraits as H 

seed = 2147483647 

io_dir = Path().cwd()/'io'
data_dir = io_dir/'data'
original_dir = data_dir/'original'
compatible_dir = data_dir/'compatible'
compatible_dir.mkdir(parents=True, exist_ok=True)

transform_dir = io_dir/'transform'
transform_dir.mkdir(parents=True, exist_ok=True)

metric_dir = io_dir/'metric'
metric_dir.mkdir(parents=True, exist_ok=True) 

color_dir = io_dir/'color'
color_dir.mkdir(parents=True, exist_ok=True)

config_dir = io_dir/'config'
config_dir.mkdir(parents=True, exist_ok=True)

model_dir = io_dir/'model'
deploy_dir = io_dir/'deploy'


##### Compatible CSVs
csv_file = Path('io/data/original/LakeViewDF.csv')
orig_df = pd.read_csv(csv_file)

orig_wave_cols, comp_wave_cols = [], []
for c in orig_df.columns:
    try:
        fc = float(c)
        orig_wave_cols.append(c)
        comp_wave_cols.append(f'X_{float(c):0.3f}')
    except:
        pass

print(f'{orig_df.shape = }, {len(orig_wave_cols) = }')

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
    orig_cols = ['sampleID'] + orig_wave_cols + [o]
    comp_cols = ['sample_id'] + comp_wave_cols + [c]
    comp_df = orig_df[orig_cols].dropna()
    comp_csv = compatible_dir/f'{c}.csv'
    comp_df.to_csv(comp_csv, index=None)
    print(f'Saved: {comp_csv.name} ({comp_df.shape})') 
print('Created compatible CSVs. \n')

##### Transforms
transforms = {'400-800-clip-uv': [H.KeepWavelengths(keep_ranges=[(399.99, 800.01)]),
                                  H.Clip(low=0.0),
                                  H.UnitVectorize()],
              '410-800-clip-uv': [H.KeepWavelengths(keep_ranges=[(409.99, 800.01)]),
                                  H.Clip(low=0.0),
                                  H.UnitVectorize()],
              '420-800-clip-uv': [H.KeepWavelengths(keep_ranges=[(419.99, 800.01)]),
                                  H.Clip(low=0.0),
                                  H.UnitVectorize()],
              '430-800-clip-uv': [H.KeepWavelengths(keep_ranges=[(429.99, 800.01)]),
                                  H.Clip(low=0.0),
                                  H.UnitVectorize()],
              '440-800-clip-uv': [H.KeepWavelengths(keep_ranges=[(439.99, 800.01)]),
                                  H.Clip(low=0.0),
                                  H.UnitVectorize()], 
              '450-800-clip-uv': [H.KeepWavelengths(keep_ranges=[(449.99, 800.01)]),
                                  H.Clip(low=0.0),
                                  H.UnitVectorize()],

              '400-800-asis-uv': [H.KeepWavelengths(keep_ranges=[(399.99, 800.01)]),
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
                                  H.UnitVectorize()]}
for (k, t) in transforms.items():
    H.save_transforms(transforms=t, transforms_file=transform_dir/f'{k}.json') 
print('Created transforms.\n')

##### Metrics
metrics = [H.R2(),
           H.RangeNormalizedRMSE(),
           H.InterquartileNormalizedRMSE(),
           H.RMSE()]
H.save_metrics(metrics=metrics, metrics_file=metric_dir/'r2-rnrmse-inrmse-rmse.json') 
print('Created metrics. \n')


##### Colors
sids = sorted(list(set(orig_df['sampleID'].values.tolist())))
color_df = DataFrame({'sample_id': sids,
                      'color': '#CC99CC'})
color_df.to_csv(color_dir/'same-color.csv', index=None)
print('Created colors. \n')


##### Train config
train_csvs = sorted([f for f in compatible_dir.glob('*.csv')])
transform_jsons = sorted([f for f in transform_dir.glob('*.json')])
    
model_type = 'plsr'
model_select = 'median-min'
n_components = 30
n_outers = 200
n_inners = 50

n_train_configs = 1
for (train_csv, transform_json) in product(train_csvs, transform_jsons):
    model_name = f'{model_type}__{model_select}__{transform_json.stem}__{train_csv.stem}-avg'
    train_config = {'seed': seed,
                    'model_name': model_name,
                    'model_type': model_type,
                    'model_dir': str(model_dir/model_name),
                    'model_selection': model_select,
                    'n_components': n_components,
                    
                    'n_outers': n_outers,
                    'n_inners': n_inners,
                    'outer_type': 'montecarlo',
                    'inner_type': 'montecarlo',
                    'test_percent': 15,
                    'valid_percent': 15, 
                              
                    'train_csv': str(train_csv),
                    'subsample': -1,
                    'reduce': 'mean',
                    'transform_json': str(transform_json)}
    
    config_json = config_dir/f'TRAIN__{model_name}.json'
    with open(config_json, 'w') as writer:
        json.dump(train_config, writer)
    print(f'Created: {config_json.stem} ({n_train_configs})')
    n_train_configs += 1
    
print('Created train configs. \n')


##### Deploy configs
n_deploy_configs = 1
for (train_csv, transform_json) in product(train_csvs, transform_jsons):
    model_name = f'{model_type}__{model_select}__{transform_json.stem}__{train_csv.stem}-avg'
    with open(config_dir/f'TRAIN__{model_name}.json', 'r') as reader:
        train_config = json.load(reader) 

    # internal eval
    deploy_csv = Path(train_config['train_csv'])
    deploy_other_cols = []
    deploy_split_label = 'TEST'
    deploy_name = f'{model_name}__{deploy_split_label}-{deploy_csv.stem}'
    deploy_config = {'deploy_dir': str(deploy_dir/deploy_name),
                     'deploy_csv': str(deploy_csv),
                     'deploy_other_cols': deploy_other_cols,
                     'deploy_subsample': -1,
                     'deploy_reduce': 'mean',
                     'deploy_split_label': deploy_split_label, 
                     'color_csv': str(color_dir/'same-color.csv'),
                     'metric_json': str(metric_dir/'r2-rnrmse-inrmse-rmse.json')} 
    deploy_config.update(train_config)
    config_json = config_dir/f'DEPLOY__{deploy_name}.json'
    with open(config_json, 'w') as writer:
        json.dump(deploy_config, writer)
    print(f'Created: {config_json.stem} ({n_deploy_configs}).')
    n_deploy_configs += 1
    
print('Created deploy configs. \n')
