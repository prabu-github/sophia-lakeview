from __future__ import annotations
import sys
from typing import List, Dict
from pathlib import Path
from collections import defaultdict
import pandas as pd
from pandas import DataFrame, Series
import argparse
import json
import shutil 
from pprint import pprint

hytraits_path = (Path(__file__).parent.parent/'hytraits').resolve()
if str(hytraits_path) not in sys.path:
    sys.path.append(str(hytraits_path))
import hytraits as H 


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
    io_dir = Path(__file__).parent/'io'
    
    return {'original': io_dir/'original',
            'compatible': io_dir/'compatible',
            'config': io_dir/'config',
            'model': io_dir/'model', 
            'deploy': io_dir/'deploy',
            'eda': io_dir/'eda'}


def get_setup_args() -> Dict:
    '''
    CLI arguments parser for setup.
    
    Return: Dict
    '''
    parser = argparse.ArgumentParser('Setup')
    parser.add_argument('--model_type', action='store', type=str, default='plsr')
    parser.add_argument('--model_select', action='store', type=str, default='median-min')
    parser.add_argument('--n_components', action='store', type=int, default=30)
    parser.add_argument('--n_outers', action='store', type=int, default=200)
    parser.add_argument('--n_inners', action='store', type=int, default=50)
    parser.add_argument('--test_percent', action='store', type=int, default=15)
    parser.add_argument('--valid_percent', action='store', type=int, default=15)
    parser.add_argument('--deploy', action='store_true', default=False)
    parser.add_argument('--eda', action='store_true', default=False)
    parser.add_argument('--cleanup', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--min_nsamples', action='store', default=1)
    return parser.parse_args().__dict__


def get_trait_args() -> Dict:
    '''
    CLI arguments parser for trait.
    
    Return: Dict
    '''
    parser = argparse.ArgumentParser('Trait')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train_config_idx', action='store', type=int)
    group.add_argument('--train_config_stem', action='store', type=str)
    group.add_argument('--eda_config_idx', action='store', type=int)
    group.add_argument('--eda_config_stem', action='store', type=str)
    parser.add_argument('--verbose', action='store_true', default=False)
    return parser.parse_args().__dict__


def get_traits() -> Dict:
    '''
    Rename traits, no space, lowercase, single words.

    Return: Dict{old: new}
    '''
    
    return dict([('IC', 'ic'),
                 ('NPOC', 'npoc'),
                 ('Cl(-)', 'cl'), 
                 ('SO4(2-)', 'so4'), 
                 ('Silica', 'si'), 
                 ('NO2/NO3', 'no23'), 
                 ('SRP', 'srp'),
                 ('NH4(+)', 'nh4'),
                 ('TN', 'tn'),
                 ('TP', 'tp'),
                 ('TSS', 'tss'),
                 ('CF_chl', 'cfchl'),
                 ('CF_PC', 'cfpc'),
                 ('PC:chl', 'pcchl'),
                 ('secchi_avg', 'secchi')])


def get_trait_transformations(transformations_file: Path) -> Dict:
    '''
    Trait transformations.

    `transformations_file`: Path
                            The transform file.
                            Columns: 'trait_name', 'transform'

    Return: Dict{trait_code:['raw', 'log']}
    '''
    PATHS = get_paths()
    TRAITS = get_traits()
    
    df = pd.read_csv(transformations_file).reset_index(drop=True)

    return {TRAITS[t]: x for (t, x) in df.values if t in TRAITS}

    
def repackage_df(df: DataFrame) -> DataFrame:
    '''
    Remaps entries in `df`; extracts required columns in `df`.
    
    `df`: DataFrame
          The dataframe from CSV file.
          
    Return: DataFrame
    '''
    TRAITS = get_traits()
    
    column_renames = TRAITS.copy()
    column_renames['sampleID'] = 'sample_id'
    for c in df.columns:
        try:
            column_renames[c] = f'X_{float(c):0.3f}'
        except:
            pass
    df.rename(columns=column_renames, inplace=True)

    wanted_cols = ['sample_id'] 
    wanted_cols += list(TRAITS.values())
    wanted_cols += [c for c in df.columns if c[:2] == 'X_']
    return df[wanted_cols]


def get_trait_indexes(df: DataFrame) -> Dict:
    '''
    `df`: DataFrame
          The repackaged DataFrame.
          Trims rows (per-trait) that are null and outside range.

    Return: Dict{str:pd.Series} ({trait:index})
            `df`.loc can be used to extract the trait's data.
    '''
    TRAITS = get_traits()
    
    trait_indexes = {}   
    for trait in TRAITS.values():
        trait_indexes[trait] = ((df[trait] > 0.0) & df[trait].notnull())

    return trait_indexes


def make_compatible_csvs(csv_file: Path,
                         verbose: bool) -> Tuple[DataFrame, DataFrame]:
    '''
    Makes trait-wise compatible CSVs.
    
    `csv_file`: Path
                Data CSV.
    `verbose`: bool
               If True, prints messages.
               
    Return: Tuple[DataFrame, DataFrame]
            Tuple[0]: repackaged DataFrame
            Tuple[1]: counts DataFrame
            Counts of the various slices.
    '''
    PATHS = get_paths()
    TRAITS = get_traits()

    # read CSV
    df = repackage_df(pd.read_csv(csv_file, low_memory=False))
    
    # get traitwise indexes
    trait_indexes = get_trait_indexes(df)

    # create compatible dir
    comp_dir = PATHS['compatible'] 
    comp_dir.mkdir(parents=True, exist_ok=True)
    
    # make data slices
    base_cols = ['sample_id']
    wave_cols = [c for c in df.columns if c[:2] == 'X_']   
    trait_counts = defaultdict(list)
    for trait in TRAITS.values():
        # extract trait slice
        wanted_cols = base_cols + [trait] + wave_cols
        trait_df = df.loc[trait_indexes[trait]][wanted_cols].dropna()
        trait_df.rename(columns={trait: 'y_true'}, inplace=True)
        trait_df.to_csv(comp_dir/f'{trait}.csv', index=None)
            
        # tracking counts for trait
        trait_counts['trait'].append(trait)
        trait_counts['n_samples'].append(trait_df['sample_id'].nunique())

    counts_df = pd.DataFrame(trait_counts)
    counts_df = counts_df[['trait', 'n_samples']]

    if verbose:
        n_csvs = len(list(PATHS['compatible'].glob('*.csv')))
        print(f'------ Created compatible CSV(s) ({n_csvs})')
            
    return (df, counts_df) 

def make_color_configs(df: DataFrame,
                       verbose: bool) -> None:
    '''
    Make color config(s).

    `df`: DataFrame
          The data table, must have `sample_id`.
    `verbose`: bool
               If True, prints messages.
    '''
    PATHS = get_paths()
    
    color_dict = {sid: '#CC99CC' for sid in df['sample_id']}
    
    PATHS['config'].mkdir(parents=True, exist_ok=True)
    
    with open(PATHS['config']/'COLOR__default.json', 'w') as writer:
        json.dump(color_dict, writer, indent=4)

    if verbose:
        n_configs = len(list(PATHS['config'].glob('COLOR__*.json')))
        print(f'------ Created color config(s) ({n_configs})')


def make_transform_configs(verbose: bool) -> None:
    '''
    Make transform config(s).

    `verbose`: bool
               If True, prints messages.
    '''
    PATHS = get_paths()

    kw_400_800 = H.KeepWavelengths(keep_ranges=[(399.99, 800.01)]) 
    kw_410_800 = H.KeepWavelengths(keep_ranges=[(409.99, 800.01)]) 
    kw_420_800 = H.KeepWavelengths(keep_ranges=[(419.99, 800.01)])
    kw_430_800 = H.KeepWavelengths(keep_ranges=[(429.99, 800.01)]) 
    kw_440_800 = H.KeepWavelengths(keep_ranges=[(439.99, 800.01)]) 
    kw_450_800 = H.KeepWavelengths(keep_ranges=[(449.99, 800.01)]) 
    uv = H.UnitVectorize()
    log_true = H.Log(apply_key='y_true')
    exp_true = H.Exp(apply_key='y_true')
    exp_pred = H.Exp(apply_key='y_pred')
    cmr = H.CommonMinReflectance()

    transforms = {'400-800-asis-raw': [kw_400_800, uv],
                  '410-800-asis-raw': [kw_410_800, uv],
                  '420-800-asis-raw': [kw_420_800, uv],
                  '430-800-asis-raw': [kw_430_800, uv],
                  '440-800-asis-raw': [kw_440_800, uv],
                  '450-800-asis-raw': [kw_450_800, uv],
                  
                  '400-800-asis-log': [kw_400_800, uv, log_true],
                  '410-800-asis-log': [kw_410_800, uv, log_true],
                  '420-800-asis-log': [kw_420_800, uv, log_true],
                  '430-800-asis-log': [kw_430_800, uv, log_true],
                  '440-800-asis-log': [kw_440_800, uv, log_true],
                  '450-800-asis-log': [kw_450_800, uv, log_true],

                  '400-800-move-raw': [kw_400_800, cmr, uv],
                  '410-800-move-raw': [kw_410_800, cmr, uv],
                  '420-800-move-raw': [kw_420_800, cmr, uv],
                  '430-800-move-raw': [kw_430_800, cmr, uv],
                  '440-800-move-raw': [kw_440_800, cmr, uv],
                  '450-800-move-raw': [kw_450_800, cmr, uv],
                  
                  '400-800-move-log': [kw_400_800, cmr, uv, log_true],
                  '410-800-move-log': [kw_410_800, cmr, uv, log_true],
                  '420-800-move-log': [kw_420_800, cmr, uv, log_true],
                  '430-800-move-log': [kw_430_800, cmr, uv, log_true],
                  '440-800-move-log': [kw_440_800, cmr, uv, log_true],
                  '450-800-move-log': [kw_450_800, cmr, uv, log_true],
                 
                  'unlog': [exp_true, exp_pred],
                  'empty': []}

    PATHS['config'].mkdir(parents=True, exist_ok=True)
    
    for (k, t) in transforms.items():
        H.save_transforms(transforms=t, 
                          transforms_file=PATHS['config']/f'TRANSFORM__{k}.json') 
    
    if verbose:
        n_configs = len(list(PATHS['config'].glob('TRANSFORM__*.json')))
        print(f'------ Created transform config(s) ({n_configs})')    


def make_metric_configs(verbose: bool) -> None:
    '''
    Make metric config(s).

    `verbose`: bool
               If True, prints messages.
    '''
    PATHS = get_paths()
    
    metrics = [H.R2(),
               H.RangeNormalizedRMSE(),
               H.InterquartileNormalizedRMSE(),
               H.RMSE()]

    PATHS['config'].mkdir(parents=True, exist_ok=True)
    
    H.save_metrics(metrics=metrics, 
                   metrics_file=PATHS['config']/'METRIC__default.json')

    if verbose:
        n_configs = len(list(PATHS['config'].glob('METRIC__*.json')))
        print(f'------ Created metric config(s) ({n_configs})')    


def cleanup() -> None:
    '''
    Cleanup project specific directories that are created,
    except for the original data directory.
    '''
    print('Cleaning up ...')
    for d in get_paths().values():
        if (d.stem != 'original') and d.exists():
            shutil.rmtree(str(d)) 


def make_train_configs(transformations_file: Path,
                       verbose: bool,
                       n_outers: int = 200,
                       n_inners: int = 50,
                       test_percent: int = 15,
                       valid_percent: int = 15) -> None:
    '''
    Make train config(s).
    `transformations_file`: Path     
                            Trait: transformation CSV.
    `verbose`: bool
               If True, prints messages.
    `n_outers`: int 
                Number outer loops.
    `n_inners`: int 
                Number inner loops.
    `test_percent`: int
                    Percentage of data for test split.
    `valid_percent`: int 
                     Percentage of data for validation split.
    
    '''
    PATHS = get_paths()
    TRAITXFORMS = get_trait_transformations(transformations_file)
    TRANSFORMS = {'raw': sorted(list(PATHS['config'].glob(f'TRANSFORM__*-raw.json'))),
                  'log': sorted(list(PATHS['config'].glob(f'TRANSFORM__*-log.json')))}

    PATHS['config'].mkdir(parents=True, exist_ok=True)
    
    model_type = 'plsr'
    train_csvs = sorted(list(PATHS['compatible'].glob('*.csv')))
    for train_csv in train_csvs:
        transforms = TRANSFORMS[TRAITXFORMS[train_csv.stem]]
        
        for transform in transforms:
            tkey = transform.stem.split('TRANSFORM__')[-1]
            model_name = f'{model_type}__{tkey}__{train_csv.stem}'

            config = H.get_config_template()
            config['seed'] = get_seed()
            config['model_name'] = model_name
            config['model_type'] = model_type
            config['model_dir'] = str(PATHS['model']/model_name)
            config['n_outers'] = n_outers
            config['n_inners'] = n_inners
            config['test_percent'] = test_percent
            config['valid_percent'] = valid_percent
            config['inner_type'] = 'montecarlo'
            config['outer_type'] = 'montecarlo'
            #-----------------------------------------
            config['train_csv'] = str(train_csv)
            config['train_subsample'] = -1
            config['train_reduce'] = 'mean'
            config['train_transform_json'] = str(transform)
            #-----------------------------------------
            config['plsr_model_selection'] = 'median-min'
            config['plsr_max_components'] = 30
            #-----------------------------------------
            H.verify_train_config(config)
            config_json = PATHS['config']/f'TRAIN__{model_name}.json'
            with open(config_json, 'w') as writer:
                json.dump(config, writer)
            
    if verbose:
        n_configs = len(list(PATHS['config'].glob('TRAIN__*.json')))
        print(f'------ Created train config(s) ({n_configs})')   


def make_eda_configs(transformations_file: Path,
                     verbose: bool) -> None:
    '''
    Make eda config(s).
    `transformations_file`: Path     
                            Trait: transformation CSV.
    `verbose`: bool
               If True, prints messages.
    '''
    PATHS = get_paths()
    TRAITXFORMS = get_trait_transformations(transformations_file)

    comp_csvs = [f for f in PATHS['compatible'].glob('*.csv')]

    for csv in comp_csvs:
        trait, tform = csv.stem, TRAITXFORMS[csv.stem]
        eda_name = f'{tform}__{trait}'
        transform_json = 'TRANSFORM__empty.json' if tform == 'raw' else 'TRANSFORM__unlog.json'
        
        config = H.get_config_template()
        config['seed'] = get_seed()
        #-----------------------------------------
        config['eda_name'] = eda_name
        config['eda_types'] = ['uni-r2',
                               'uni-pearson-correlation',
                               'ndi-r2',
                               'ndi-pearson-correlation']
        config['eda_dir'] = str(PATHS['eda']/eda_name)
        config['eda_csv'] = str(csv)
        config['eda_subsample'] = -1
        config['eda_reduce'] = 'mean'
        config['eda_transform_json'] = str(PATHS['config']/transform_json)
        #-----------------------------------------
        H.verify_eda_config(config)
        config_json = PATHS['config']/f'EDA__{eda_name}.json'
        with open(config_json, 'w') as writer:
            json.dump(config, writer)
                
    if verbose:
        n_configs = len(list(PATHS['config'].glob('EDA__*.json')))
        print(f'------ Created eda config(s) ({n_configs})')


def make_deploy_configs(verbose: bool) -> None:
    '''
    Make deploy config(s).

    `verbose`: bool
               If True, prints messages.
    '''
    PATHS = get_paths()

    train_config_jsons = [f for f in PATHS['config'].glob('TRAIN__*.json')]
    train_config_jsons.sort()

    deploy_split_label = 'TEST'
    deploy_subsample = -1
    deploy_reduce = 'mean'
    for train_config_json in train_config_jsons:
        # load train config; extract details
        with open(train_config_json, 'r') as reader:
            train_config = json.load(reader)
        model_name = train_config['model_name']
        transform_type = model_name.split('__')[1]
        if transform_type == 'raw':
            transform_json = PATHS['config']/'TRANSFORM__empty.json'
        else:
            transform_json = PATHS['config']/'TRANSFORM__unlog.json'
        deploy_csv = Path(train_config['train_csv'])
        deploy_name = f'{model_name}__{deploy_split_label}__{deploy_csv.stem}'

        config = H.get_config_template()
        config.update(train_config)
        #-----------------------------------------
        config['deploy_dir'] = str(PATHS['deploy']/deploy_name)
        config['deploy_csv'] = str(deploy_csv)
        config['deploy_other_cols'] = []
        config['deploy_subsample'] = deploy_subsample
        config['deploy_reduce'] = deploy_reduce
        config['deploy_split_label'] = deploy_split_label
        config['deploy_transform_json'] = str(transform_json)
        config['deploy_color_json'] = str(PATHS['config']/'COLOR__default.json')
        config['deploy_metric_json'] = str(PATHS['config']/'METRIC__default.json')
        #-----------------------------------------
        H.verify_deploy_config(config)
        config_json = PATHS['config']/f'DEPLOY__{deploy_name}.json'
        with open(config_json, 'w') as writer:
            json.dump(config, writer)
    
    if verbose:
        n_configs = len(list(PATHS['config'].glob('DEPLOY__*.json')))
        print(f'------ Created deploy config(s) ({n_configs})')


def get_train_config_json(train_config_idx: int = None,
                          train_config_stem: str = None) -> Path:
    '''
    TRAIN__ JSON corresponing to `train_config_idx` in sorted or
    TRAIN__ JSON corresponing to `train_config_stem`

    `train_config_idx`: int 
                        Index into sorted list of TRAIN__ config files.
    `train_config_stem`: int 
                         Stem of a particular TRAIN__ config file.
                        
    Return: Path
            If `train_config_idx` and `train_config_stem` are both None, None is returned.
    '''
    if (train_config_idx is None) and (train_config_stem is None):
        return None
        
    if (train_config_idx is not None) and (train_config_stem is not None):
        raise Exception('One of train_config_idx/train_config_stem to be specified; not both.')
        
    PATHS = get_paths()

    train_jsons = [f for f in PATHS['config'].glob('TRAIN__*.json')]
    train_jsons.sort()

    if train_config_idx is not None:
        if (train_config_idx < 0) or (train_config_idx >= len(train_jsons)):
            raise Exception(f'{train_config_idx} is invalid.')
        return train_jsons[train_config_idx]
    else:
        found = False
        for f in train_jsons:
            if f.stem == train_config_stem:
                return f
        if (not found):
            raise Exception(f'{train_config_stem} is not found.')


def get_deploy_config_jsons(train_config_idx: int = None,
                            train_config_stem: str = None) -> List[Path]:
    ''' 
    DEPLOY__ JSON corresponing to `train_config_idx` in sorted, or
    DEPLOY__ JSON corresponing to `train_config_stem`

    `train_config_idx`: int 
                        Index into sorted list of TRAIN__ config files.
    `train_config_stem`: int 
                         Stem of a particular TRAIN__ config file.

    Return: List[Path]  
            If `train_config_idx` and `train_config_stem` are both None, None is returned.
    '''
    PATHS = get_paths()

    train_json = get_train_config_json(train_config_idx=train_config_idx,
                                       train_config_stem=train_config_stem)
    if train_json is not None:
        train_name = train_json.stem
        model_name = '__'.join(train_name.split('__')[1:])
    
        all_deploy_jsons = [f for f in PATHS['config'].glob('DEPLOY__*.json')]
        deploy_jsons = [f for f in all_deploy_jsons if model_name in f.stem]
        deploy_jsons.sort()
    
        return deploy_jsons
    else:
        return None


def get_eda_config_json(eda_config_idx: int = None,
                        eda_config_stem: str = None) -> Path:
    '''
    EDA__ JSON corresponing to `eda_config_idx` in sorted.

    `eda_config_idx`: int 
                      Index into sorted list of EDA__ config files.
    `eda_config_stem`: int 
                       Stem of a particular EDA__ config file.
                        
    Return: Path
            If `eda_config_idx` and `eda_config_stem` are both None, None is returned.
    '''
    if (eda_config_idx is None) and (eda_config_stem is None):
        return None
        
    if (eda_config_idx is not None) and (eda_config_stem is not None):
        raise Exception('One of eda_config_idx/eda_config_stem to be specified; not both.')
        
    PATHS = get_paths()

    eda_jsons = [f for f in PATHS['config'].glob('EDA__*.json')]
    eda_jsons.sort()

    if eda_config_idx is not None:
        if (eda_config_idx < 0) or (eda_config_idx >= len(eda_jsons)):
            raise Exception(f'{eda_config_idx} is invalid.')
        return eda_jsons[eda_config_idx]
    else:
        found = False
        for f in eda_jsons:
            if f.stem == eda_config_stem:
                found = True
                return f
        if (not found):
            raise Exception(f'{eda_config_stem} is not found.')
