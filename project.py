from __future__ import annotations
from typing import List, Dict, Tuple, Union
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from pandas import DataFrame
import json
from itertools import product, chain
from collections import defaultdict
import fnmatch
from pprint import pprint
from scipy.stats import yeojohnson

hytraits_path = (Path(__file__).parent.parent/'hytraits').resolve()
if str(hytraits_path) not in sys.path:
    sys.path.append(str(hytraits_path))
import hytraits as H 

from utils import (get_paths, 
                   get_seed)

    
def get_xtransforms(mn: str = None) -> Dict:
    '''
    xtransform strategies. 

    Return: Dict
            {xtransform_key: List[H.BaseTransform]}        
    '''
    # [400-800] X [uv, d1, wr] X [with Yeo-Johnson, without Yeo-Johnson] 
    tgt_centers = np.arange(405, 801, 5, dtype=float)
    wrs = H.WavelengthResampling(target_centers=tgt_centers,
                                 target_fwhms=None)
    
    vsbl = H.KeepWavelengths(keep_ranges=[(400.0, 800.0)])
    uv = H.UnitVectorize()
    d1 = H.SavitzkyGolay(deriv=1)
    
    xtransforms = {'uv': [vsbl, uv],
                   'd1': [vsbl, wrs, d1],
                   'wr': [vsbl, wrs, uv]}

    if mn is None:
        return xtransforms
    else:
        k_xt = mn.split('__')[2]
        return {k_xt: xtransforms[k_xt]}

    
def get_ytransforms(mn: str = None) -> Dict:
    '''
    ytransform strategies. 

    `mn`: str
          Model name.

    Return: Dict
            {ytransform_key: List[H.BaseTransform]}        
    '''
    ytransforms = {'id': [],
                   'sqrt': [H.Log10Transform()],
                   'log10': [H.SqrtTransform()]}
    if mn is None:
        return ytransforms
    else:
        k_yt = mn.split('__')[3]
        return {k_yt: ytransforms[k_yt]}

    
def get_subsamplings() -> List[str]:
    '''
    Subsampling strategies.

    Return: List[str]
            Options: ['avg', 'rep']
    '''
    return ['rep']


def get_final_selects() -> List[str]:
    '''
    Final selection strategies

    Return: List[str]
            Options: ['boa', 'bpl']
    '''
    return ['boa']


def get_model_types() -> List[str]:
    '''
    Model types to consider.

    Return: List[str]
    '''
    return ['plsr']


def get_model_suffixes() -> List[str]:
    '''
    List of model sufffixes.

    Return: List[str]
    Product of xtrans, ytrans, subsampling, final selection.
    <xtrans>__<ytrans>__<subsample>__<finalselection>.
    Used in model name as: <trait>__<suffix>. 
    '''
    xtrans = sorted(list(get_xtransforms().keys()))
    ytrans = sorted(list(get_ytransforms().keys()))
    subsamples = sorted(get_subsamplings())
    finals = sorted(get_final_selects())
    parts = [xtrans, ytrans, subsamples, finals]
    suffixes = ['__'.join(t) for t in product(*parts)]
    return suffixes

def get_trait_dfs(data_csv) -> Dict:
    PATHS = get_paths()
    orig_dir = PATHS['original']

    full_df = pd.read_csv(orig_dir/data_csv)
    pieces = [full_df['sample_id'].astype(str),
              full_df['date'].astype(str),
              full_df['event'].astype(str),
              full_df['site'].astype(str),
              map(str, np.arange(full_df.shape[0]))]
    full_df['unique_id'] = [':'.join(t) for t in zip(*pieces)]
    
    wave_cols = []
    for c in full_df.columns:
        try:
            w = float(c)
            wave_cols.append(w)
        except:
            pass
    wave_remap = {str(wc): f'X_{float(wc):4.4f}' for wc in wave_cols}
    full_df = full_df.rename(columns=wave_remap)
    
    trait_dfs = {}
    trait_cols = ['temp', 
                  'mmHg',
                  'DO_pct',
                  'DO_conc', 
                  'SPcond', 
                  'cond', 
                  'IC', 
                  'NPOC', 
                  'chloride', 
                  'sulfate',
                  'Silica', 
                  'nitrite_nitrate', 
                  'SRP', 
                  'ammonium', 
                  'TN', 
                  'TP', 
                  'TSS',
                  'chl_CF', 
                  'phyco_CF', 
                  'ratio_phyco_chl', 
                  'chla', 
                  'chlb', 
                  'chlc',
                  'secchi_avg']
    wave_cols = [c for c in full_df.columns if c[:2] == 'X_']
    reqd_cols = ['sample_id', 'unique_id']
    for tt in trait_cols:
        want_cols = reqd_cols + [tt] + wave_cols
        tdf = full_df[want_cols].dropna()
        tdf = tdf.rename(columns={tt:'y_true'})
        trait_dfs[f'{tt}.csv'] = tdf
    return trait_dfs

    
def make_compatible_csvs() -> None:
    PATHS = get_paths()
    comp_dir = PATHS['compatible']
    comp_dir.mkdir(parents=True, exist_ok=True) 

    comp_csvs = {}
    data_csv = 'LakeView_hytraitsinput_20260424.csv' 
    trait_dfs = get_trait_dfs(data_csv=data_csv)
    for (tt, tdf) in trait_dfs.items():
        if tdf.shape[0] > 0:
            tdf.to_csv(comp_dir/f'{tt}', index=False)
            print(f'Saved: {tt}, {tdf["sample_id"].nunique()}')
        comp_csvs[tt] = int(tdf['sample_id'].nunique())
        
    with open(comp_dir/'compatible_details.json', 'w') as writer:
        json.dump(comp_csvs, writer, indent=4)


def get_model_csvs(model_name: str,
                   min_n_samples: int) -> Tuple[List[str], Dict]:
    '''
    CSVs for model training and deployment for 
    specified model.

    `model_name`: str
                  Model name string.
    `min_n_samples`: int
                     Minimum n_samples for model training.

    Return: Tuple[List[str], Dict]
            Tuple[0]: List of CSVs for model training dataset
                      If empty, then this model cannot be trained.
            Tuple[1]: Dict
                      {deploy_key: List[comp_csv_filenames]}
    '''
    tkns = model_name.split('__')
    tt = tkns[1]
   
    comp_keys = [f'{tkns[1]}.csv']
    PATHS = get_paths()
    comp_dir = PATHS['compatible']
    
    with open(comp_dir/'compatible_details.json', 'r') as reader:
        comp_n_sids = json.load(reader)
    
    train_csvs, deploy_csvs = [], defaultdict(list)
    n_sids = 0
    for comp_key in comp_keys:
        if comp_n_sids[comp_key] > 0:
            train_csvs.append(comp_key)
            n_sids += comp_n_sids[comp_key]

    # if n_sids from the compatible CSVs do not
    # have required minimum n_samples, no model to train!!
    if n_sids < min_n_samples:
        train_csvs = []
        return (train_csvs, deploy_csvs)
    
    return train_csvs, deploy_csvs


def get_model_names(min_n_samples: int,
                    pattern: str) -> List[str]:
    '''
    Models with training datasets with >= `min_n_samples`
    Filters names that match `pattern`

    Return: List[str]
            List of model names that match pattern and condition.
            <model>__<trait>__<xtrans>__<ytrans>__<subsample>__<final>
            <trait>: <meas>-<traitcol>-<proj_key>-<year_key>
    '''
    PATHS = get_paths()
    comp_dir = PATHS['compatible']
    models = get_model_types()
    suffixes = get_model_suffixes()
    with open(comp_dir/'compatible_details.json', 'r') as reader:
        d = json.load(reader)
    traits = [Path(k).stem for (k, v) in d.items() if v >= min_n_samples]
    
    model_names = []
    pieces = [models, traits, suffixes]
    for (a, b, c) in product(*pieces):
        mn = f'{a}__{b}__{c}'    
        (train_csvs, _) = get_model_csvs(model_name=mn,
                                         min_n_samples=min_n_samples)
        if len(train_csvs) > 0:
            model_names.append(mn)
        
    return sorted(fnmatch.filter(model_names, pattern))

        
if __name__ == '__main__':
    print('Sri Ramajayam')
    print()

    # make_compatible_csvs()
    
    suffixes = get_model_suffixes()
    pprint(suffixes)

    model_names = get_model_names(min_n_samples=30,
                                  pattern='*')
    pprint(model_names)
    print(f'{len(model_names) = }')
    
    xts = get_xtransforms()
    for (k, v) in xts.items():
        print(k)
        for t in v:
            pprint(t.get_name_params())
        print()
    print('--------------------')
    
    xts = get_xtransforms('plsr__ratio_phyco_chl__uv__log10__rep__boa')
    for (k, v) in xts.items():
        print(k)
        for t in v:
            pprint(t.get_name_params())
        print()
    xts = get_xtransforms('plsr__ratio_phyco_chl__wr__id__rep__boa')
    for (k, v) in xts.items():
        print(k)
        for t in v:
            pprint(t.get_name_params())
        print()

    print('=====================\n')
    
    yts = get_ytransforms()
    for (k, v) in yts.items():
        print(k)
        for t in v:
            pprint(t.get_name_params())
        print()
    print('--------------------')
    
    yts = get_ytransforms('plsr__ratio_phyco_chl__uv__log10__rep__boa')
    for (k, v) in yts.items():
        print(k)
        for t in v:
            pprint(t.get_name_params())
        print()
        
    yts = get_ytransforms('plsr__ratio_phyco_chl__uv__sqrt__rep__boa')
    for (k, v) in yts.items():
        print(k)
        for t in v:
            pprint(t.get_name_params())
        print()

    yts = get_ytransforms('plsr__ratio_phyco_chl__uv__id__rep__boa')
    for (k, v) in yts.items():
        print(k)
        for t in v:
            pprint(t.get_name_params())
        print()
    
    