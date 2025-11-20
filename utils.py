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
            'ideploy': io_dir/'ideploy',
            'edeploy': io_dir/'edeploy',
            'eda': io_dir/'eda'}


def get_setup_args(message: str) -> Dict:
    '''
    CLI arguments parser for traitmodleing.
    
    `message`: str
    
    Return: Dict
    '''
    parser = argparse.ArgumentParser(message)
    parser.add_argument('--model_type', action='store', type=str, default='plsr')
    parser.add_argument('--model_select', action='store', type=str, default='median-min')
    parser.add_argument('--n_components', action='store', type=int, default=30)
    parser.add_argument('--n_outers', action='store', type=int, default=200)
    parser.add_argument('--n_inners', action='store', type=int, default=50)
    parser.add_argument('--test_percent', action='store', type=int, default=15)
    parser.add_argument('--valid_percent', action='store', type=int, default=15)
    parser.add_argument('--ideploy', action='store_true', default=False)
    parser.add_argument('--edeploy', action='store_true', default=False)
    parser.add_argument('--eda', action='store_true', default=False)
    parser.add_argument('--cleanup', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--min_nsamples', action='store', default=1)
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

    transforms = {'400-800-asis-uv-raw': [kw_400_800, uv],
                  '410-800-asis-uv-raw': [kw_410_800, uv],
                  '420-800-asis-uv-raw': [kw_420_800, uv],
                  '430-800-asis-uv-raw': [kw_430_800, uv],
                  '440-800-asis-uv-raw': [kw_440_800, uv],
                  '450-800-asis-uv-raw': [kw_450_800, uv],
                  
                  '400-800-asis-uv-log': [kw_400_800, uv, log_true],
                  '410-800-asis-uv-log': [kw_410_800, uv, log_true],
                  '420-800-asis-uv-log': [kw_420_800, uv, log_true],
                  '430-800-asis-uv-log': [kw_430_800, uv, log_true],
                  '440-800-asis-uv-log': [kw_440_800, uv, log_true],
                  '450-800-asis-uv-log': [kw_450_800, uv, log_true],

                  '400-800-move-uv-raw': [kw_400_800, cmr, uv],
                  '410-800-move-uv-raw': [kw_410_800, cmr, uv],
                  '420-800-move-uv-raw': [kw_420_800, cmr, uv],
                  '430-800-move-uv-raw': [kw_430_800, cmr, uv],
                  '440-800-move-uv-raw': [kw_440_800, cmr, uv],
                  '450-800-move-uv-raw': [kw_450_800, cmr, uv],
                  
                  '400-800-move-uv-log': [kw_400_800, cmr, uv, log_true],
                  '410-800-move-uv-log': [kw_410_800, cmr, uv, log_true],
                  '420-800-move-uv-log': [kw_420_800, cmr, uv, log_true],
                  '430-800-move-uv-log': [kw_430_800, cmr, uv, log_true],
                  '440-800-move-uv-log': [kw_440_800, cmr, uv, log_true],
                  '450-800-move-uv-log': [kw_450_800, cmr, uv, log_true],
                 
                  'undo-log': [exp_true, exp_pred]}

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


def make_train_configs(transformations_file: Path,
                       verbose: bool) -> None:
    '''
    Make train config(s).
    `transformations_file`: Path
                            
    `verbose`: bool
               If True, prints messages.
    '''
    PATHS = get_paths()
    TXFORMS = get_trait_transformations(transformations_file)
    
    if verbose:
        n_configs = len(list(PATHS['config'].glob('TRAIN__*.json')))
        print(f'------ Created train config(s) ({n_configs})')   


def make_ideploy_configs(verbose: bool) -> None:
    '''
    Make ideploy config(s).

    `verbose`: bool
               If True, prints messages.
    '''
    PATHS = get_paths()

    if verbose:
        n_configs = len(list(PATHS['config'].glob('IDEPLOY__*.json')))
        print(f'------ Created ideploy config(s) ({n_configs})')


def make_edeploy_configs(verbose: bool) -> None:
    '''
    Make edeploy config(s).

    `verbose`: bool
               If True, prints messages.
    '''
    PATHS = get_paths()

    if verbose:
        n_configs = len(list(PATHS['config'].glob('EDEPLOY__*.json')))
        print(f'------ Created edeploy config(s) ({n_configs})')


def make_eda_configs(verbose: bool) -> None:
    '''
    Make eda config(s).

    `verbose`: bool
               If True, prints messages.
    '''
    PATHS = get_paths()

    if verbose:
        n_configs = len(list(PATHS['config'].glob('EDA__*.json')))
        print(f'------ Created eda config(s) ({n_configs})')

        
def cleanup() -> None:
    '''
    Cleanup project specific directories that are created,
    except for the original data directory.
    '''
    print('Cleaning up ...')
    for d in get_paths().values():
        if (d.stem != 'original') and d.exists():
            shutil.rmtree(str(d))    
