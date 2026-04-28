from __future__ import annotations
from typing import List, Dict, Tuple, Union
import sys
from pathlib import Path
import numpy as np
from pprint import pprint

hytraits_path = (Path(__file__).parent.parent/'hytraits').resolve()
if str(hytraits_path) not in sys.path:
    sys.path.append(str(hytraits_path))
import hytraits as H 

from utils import (get_cli_args,
                   get_paths,
                   get_seed)
from project import (make_compatible_csvs,
                     get_model_names,
                     get_xtransforms,
                     get_ytransforms,
                     get_model_csvs)

DatasetAndSplits = Tuple[H.CSVDataset, H.CSVDatasetSplits]


def get_dataset_splits(csvs: List[str],
                       # suffix: str,
                       model_name: str,
                       splits_needed: bool,
                       n_repeats: Tuple[int, int],
                       split_types: Tuple[str, str],
                       split_percents: Tuple[int, int],
                       purpose: str) -> DatasetAndSplits:
    '''
    Provides dataset and splits.

    `csvs`: List[str]
            List of CSV filenames. 
            Parents are added in function.
    `suffix`: str
              Model's suffix part.
              <xtrans>__<ytrans>__<subsample>__<finalselection>
    `splits_needed`: bool
                    If False, splits is None.
    `n_repeats`: Tuple[int, int]
                 Repeats for (outer, inner) loops.
    `split_types`: Tuple[str, str]
                   Types of (outer, inner) loops.
    `split_percents`: Tuple[int, int]
                      Percentages for (test, valid) sets.
                      Percentage of entire dataset.
    `purpose`: str
               The purpose for the data.
               Options: ['train', 'deploy']
               Decides subsample and reduce params.
    
    Return: DatasetAndSplits
            (CSVDataset, CSVDatasetSplits)
    '''
    PATHS = get_paths()
    comp_dir = PATHS['compatible']
    
    # xt, yt, ss, fs = suffix.split('__')
    mm, tt, xt, yt, ss, fs = model_name.split('__')  
    xtransforms = get_xtransforms(mn)[xt]
    if purpose == 'deploy':
        subsample = -1
        reduce = 'none'
        ytransforms = []
    else:
        subsample = 1 if (ss == 'rep') else -1
        reduce = 'none' if (ss == 'rep') else 'mean'
        ytransforms = get_ytransforms(mn)[yt]
        
    
    dataset = H.CSVDataset(csv_file=[comp_dir/f for f in csvs],
                           xtransforms=xtransforms,
                           ytransforms=ytransforms,
                           subsample=subsample,
                           reduce=reduce,
                           seed=get_seed()) 

    splits = None
    if splits_needed:
        splitter = H.CSVDatasetSplitter(n_repeats=n_repeats,
                                        split_types=split_types,
                                        split_percents=split_percents,
                                        seed=get_seed())
        splits = splitter(dataset.sids)

    return (dataset, splits)


def train_model(model_name: str,
                csvs: List[str],
                n_repeats: Tuple[int, int],
                split_types: Tuple[str, str],
                split_percents: Tuple[int, int],
                params: Dict) -> None:
    '''
    Trains the model. 
    Handles dataset creation, training, saving.
    Creates: io_/model/model_name

    `model_name`: str
                  Name of model.
                  <model>__<trait>__<xtrans>__<ytrans>__<subsample>__<final>
    `csvs`: List[str]
            List of CSV filenames. 
            Parents are added in function.
    `n_repeats`: Tuple[int, int]
                 Repeats for (outer, inner) loops.
    `split_types`: Tuple[str, str]
                   Types of (outer, inner) loops.
    `split_percents`: Tuple[int, int]
                      Percentages for (test, valid) sets.
                      Percentage of entire dataset.
    `params`: Dict
              Useful stuff.
              eg. n_components can be specified here for PLSR.
    '''
    print(f'Training: {model_name}')
    
    PATHS = get_paths()
    model_dir = PATHS['model']/model_name
    
    model_type = model_name.split('__')[0]
    (dataset, splits) = get_dataset_splits(csvs=csvs,
                                           model_name=model_name,
                                           n_repeats=n_repeats,
                                           split_types=split_types,
                                           split_percents=split_percents,
                                           purpose='train',
                                           splits_needed=True)
    # data = dataset[None]
    # pprint(data['y_true'][:3])
    # pprint(data['unique_id'][:3])
    
    if model_type == 'plsr':
        final = model_name.split('__')[-1]
        n_train = len(splits.split(0, 0, 'train'))
        ncs = min(params['n_components'], n_train - 2)
        n_comps = np.arange(1, (ncs + 1)).tolist()
        H.plsr_train(dataset=dataset,
                     splits=splits,
                     save_dir=model_dir,
                     n_components=n_comps,
                     final=final)
        H.plsr_npz_to_json(model_file=model_dir/'model.npz')
    else:
        raise Exception(f'{model_type} is invalid.')
        

def deploy_model(model_name: str,
                 deploy_key: str,
                 deploy_csvs: List[str],
                 splits_needed: bool,
                 n_repeats: Tuple[int, int] = (200, 30),
                 split_types: Tuple[str, str] = ('montecarlo', 'montecarlo'),
                 split_percents: Tuple[int, int] = (15, 15),
                 split_label: str = 'test') -> None:
    '''
    Deploys specified model on provided CSVs.

    `model_name`: str
                  Model to deploy.
                  io_/model/model_name must be populated already.
                  <model>__<trait>__<xtrans>__<ytrans>__<subsample>__<final>
    `deploy_key`: str
                  Deployment key
                  If None, io_/deploy/model_name created
                  Else, io_/deploy/model_name__ON__deploy_key created
    `deploy_csvs`: List[str]
                   List of CSV filenames for dataset. 
                   Parents are added in function.
    `splits_needed`: bool
                     If True, index_matched deployment.
                     If False, many on one deployment.
    `n_repeats`: Tuple[int, int]
                 Repeats for (outer, inner) loops.
    `split_types`: Tuple[str, str]
                   Types of (outer, inner) loops.
    `split_percents`: Tuple[int, int]
                      Percentages for (test, valid) sets.
                      Percentage of entire dataset.
    `split_label`: str
                   Split to use is `splits_needed` is True.         
    '''
    if deploy_key is None:
        deploy_name = model_name
    else:
        deploy_name = f'{model_name}__ON__{deploy_key}'
    print(f'Deploying: {deploy_name}')

    PATHS = get_paths()
    comp_dir = PATHS['compatible']
    model_dir = PATHS['model']/model_name
    deploy_dir = PATHS['deploy']/deploy_name
    
    model_type = model_name.split('__')[0]
    if model_type == 'plsr':
        model = H.plsr_load_model(model_dir/'model.npz')
    else:
        raise Exception(f'{model_type} is invalid.')

    (dataset, splits) = get_dataset_splits(csvs=deploy_csvs,
                                           model_name=model_name,
                                           n_repeats=n_repeats,
                                           split_types=split_types,
                                           split_percents=split_percents,
                                           purpose='deploy',
                                           splits_needed=splits_needed)
    # data = dataset[None]
    # pprint(data['y_true'][:3])
    # pprint(data['unique_id'][:3])
    
    metrics = [H.RMSE(),
               H.RangeNormalizedRMSE(),
               H.InterquartileNormalizedRMSE(),
               H.R2(),
               H.FittedR2()]
    undos = H.load_transforms(model_dir/'ytransforms.json',
                              undo=True)

    if not splits_needed:
        (pdf, mdf) = H.deploy_csv_many_on_one(model=model,
                                              metrics=metrics,
                                              dataset=dataset,
                                              undos=undos)
    else:
        (pdf, mdf) = H.deploy_csv_index_matched(model=model,
                                                metrics=metrics,
                                                dataset=dataset,
                                                splits=splits,
                                                split_label=split_label,
                                                undos=undos)

    deploy_dir.mkdir(parents=True, exist_ok=True)
    pdf.to_csv(deploy_dir/'preds.csv', index=False)
    mdf.to_csv(deploy_dir/'metrics.csv', index=False)

    if mdf.shape[0] > 0:
        mcols = [c for c in mdf.columns if c != 'model_idx']
        
        df_mean = mdf[mcols].mean().to_frame().T
        df_mean['deploy_name'] = deploy_name 
        df_mean = df_mean[['deploy_name'] + mcols]
        df_mean['y_true_min'] = float(model.get('y_true_min'))
        df_mean['y_true_max'] = float(model.get('y_true_max'))
        df_mean['n_samples'] = int(model.get('n_samples'))
        df_mean['n_calib'] = int(model.get('n_calib'))
        df_mean['n_deploy'] = int(pdf['sample_id'].nunique())

        if model_type == 'plsr':
            df_mean['n_components'] = int(model.get('n_components').flatten()[0])
            
        df_mean.to_csv(deploy_dir/'metrics_mean.csv', index=False)
        
    
def do_model(model_name: str,
             min_n_samples: int,
             params: Dict,
             n_repeats: Tuple[int, int] = (200, 30),
             split_types: Tuple[str, str] = ('montecarlo', 'montecarlo'),
             split_percents: Tuple[int, int] = (15, 15)) -> None:
    '''
    Build and deploy the specified model.
    '''  
    train_csvs, deploy_csvs = get_model_csvs(model_name=model_name,
                                             min_n_samples=min_n_samples)
    
    # train
    train_model(model_name=model_name,
                csvs=train_csvs,
                n_repeats=n_repeats,
                split_types=split_types,
                split_percents=split_percents,
                params=params)

    # internal eval
    deploy_model(model_name=model_name,
                 deploy_key=None,
                 deploy_csvs=train_csvs,
                 splits_needed=True,
                 n_repeats=n_repeats,
                 split_types=split_types,
                 split_percents=split_percents,
                 split_label='test')

    # external eval
    for (key, csvs) in deploy_csvs.items():
        deploy_model(model_name=model_name,
                     deploy_key=key,
                     deploy_csvs=csvs,
                     splits_needed=False,
                     n_repeats=n_repeats,
                     split_types=split_types,
                     split_percents=split_percents,
                     split_label='test')


if __name__ == '__main__':
    ARGS = get_cli_args()
    min_n_samples = ARGS['min_n_samples']
    pattern = ARGS['pattern']
    n_repeats = ARGS['n_repeats']
    split_types = ARGS['split_types']
    split_percents = ARGS['split_percents']
    params = {'n_components': ARGS['n_components']}
        
    if ARGS['make_compatible']:
        make_compatible_csvs()
    else:
        model_names = get_model_names(min_n_samples=min_n_samples,
                                      pattern=pattern)
        for mn in model_names:
            do_model(model_name=mn,
                     min_n_samples=min_n_samples,
                     params=params,
                     n_repeats=n_repeats,
                     split_types=split_types,
                     split_percents=split_percents)