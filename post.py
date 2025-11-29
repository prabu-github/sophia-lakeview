from __future__ import annotations
from typing import List, Dict, Tuple
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from pandas import DataFrame
import json
from collections import defaultdict
from itertools import product
from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import argparse

hytraits_path = (Path.cwd().parent/'hytraits').resolve()
if str(hytraits_path) not in sys.path:
    sys.path.append(str(hytraits_path))
import hytraits as H 
from utils import (get_paths,
                   get_traits)

def get_deploy_by_trait() -> Dict:
    '''
    Return: Dict
            Key: str; trait name
            Value: Dict; {str (treatment):, Path (deploy dir)}
    '''
    PATHS['post'].mkdir(parents=True, exist_ok=True)

    deploy_dirs = [e for e in PATHS['deploy'].glob('*') if e.is_dir()]
    deploy_dirs.sort()

    out = defaultdict(dict)
    for deploy_dir in deploy_dirs:
        model, treatment, train_on, split, test_on = deploy_dir.stem.split('__')
        out[train_on][treatment] = deploy_dir
        
    return out


def get_6x6_trait_deploy(trait_deploy: Dict) -> List[Tuple[str, List[Path]]]:
    '''
    Returns directories suitable for 6x6 grid plotting.

    `trait_deploy`: Dict; {str (treatment):, Path (deploy dir)}

    Return: List[Tuple[str, List[Path]]]
            List[r]: corresponds to r-th row
            Tuple[str, List[Path]]: Tuple[0] - treatment
                                    Tuple[1][c] - deploy dir for c-th column
    '''
    suffixes = ['asis-raw', 'asis-log', 'move-raw', 'move-log', 'move-pa-raw', 'move-pa-log']
    starts = [400, 410, 420, 430, 440, 450]
    d = defaultdict(list)
    for suffix in suffixes:
        for start in starts:
            key = f'{start}-{suffix}'
            d[suffix].append(trait_deploy[key])
    return [(suffix, sorted(d[suffix])) for suffix in suffixes]

    
def get_model_by_trait() -> Dict:
    '''
    Return: Dict
            Key: str; trait name
            Value: Dict; {str (treatment):, Path (model dir)}
    '''
    PATHS['post'].mkdir(parents=True, exist_ok=True)

    model_dirs = [e for e in PATHS['model'].glob('*') if e.is_dir()]
    model_dirs.sort()

    out = defaultdict(dict)
    for model_dir in model_dirs:
        model, treatment, train_on = model_dir.stem.split('__')
        out[train_on][treatment] = model_dir
        
    return out


def get_6x6_trait_model(trait_model: Dict) -> List[Tuple[str, List[Path]]]:
    '''
    Returns directories suitable for 6x6 grid plotting.

    `model_deploy`: Dict; {str (treatment):, Path (model dir)}

    Return: List[Tuple[str, List[Path]]]
            List[r]: corresponds to r-th row
            Tuple[str, List[Path]]: Tuple[0] - treatment
                                    Tuple[1][c] - model dir for c-th column
    '''
    suffixes = ['asis-raw', 'asis-log', 'move-raw', 'move-log', 'move-pa-raw', 'move-pa-log']
    starts = [400, 410, 420, 430, 440, 450]
    d = defaultdict(list)
    for suffix in suffixes:
        for start in starts:
            key = f'{start}-{suffix}'
            d[suffix].append(trait_model[key])
    return [(suffix, sorted(d[suffix])) for suffix in suffixes]


def get_eda_by_trait() -> Dict:
    '''
    EDA directories per trait.

    Return: Dict; {str (trait): List[Tuple[str, Path]] ([(treatment, eda dir)])}
    '''
    TRAITS = get_traits()
    
    suffixes = ['asis-raw', 'asis-log', 'move-raw', 'move-log', 'move-pa-raw', 'move-pa-log']
    d = defaultdict(list)
    for trait in TRAITS.values():
        for suffix in suffixes:
            d[trait].append((suffix, PATHS['eda']/f'{trait}-{suffix}-eda'))
    return d

    
def get_ordered_treatments() -> List[str]:
    '''
    Return: List[str]
            List of treatments in desired order.
            start-asis-raw (x6)
            start-asis-log (x6)
            start-move-raw (x6) 
            start-move-log (x6)
            start-move-pa-raw (x6)
            start_move-pa-log (x6)
    '''
    ordered = []
    suffixes = ['asis-raw', 'asis-log', 'move-raw', 'move-log', 'move-pa-raw', 'move-pa-log']
    for suffix in suffixes:
        for start in [400, 410, 420, 430, 440, 450]:
            ordered.append(f'{start}-{suffix}')
    return ordered


start_colors = ['#EF476F', '#F78C6B', '#FFD166', '#06D6A0', '#118A32', '#073B4C']

def make_metrics() -> None:
    PATHS = get_paths()
    TRAITS = get_traits()
    trait_deploys = get_deploy_by_trait()
    treatments = get_ordered_treatments()
    idxs = np.arange(len(treatments))
    trait_metrics_dfs = []

    print('Metrics ...')
    for trait in TRAITS.values():
        print(f'--- Making {trait}_metrics.jpg, {trait}_metrics.csv')
        trait_deploy = trait_deploys[trait]
        mean_r2s, mean_rmses, mean_rnrmses, mean_qnrmses = [], [], [], []
        sdev_r2s, sdev_rmses, sdev_rnrmses, sdev_qnrmses = [], [], [], []
        for treatment in treatments:
            metrics_df = pd.read_csv(trait_deploy[treatment]/'metrics.csv')
            column_means = metrics_df.mean(numeric_only=True)
            column_sdevs = metrics_df.std(numeric_only=True)
            
            mean_r2s.append(column_means.loc['r2'])
            mean_rmses.append(column_means.loc['rmse'])
            mean_rnrmses.append(column_means.loc['range_normalized_rmse'])
            mean_qnrmses.append(column_means.loc['interquartile_normalized_rmse'])
            sdev_r2s.append(column_sdevs.loc['r2'])
            sdev_rmses.append(column_sdevs.loc['rmse'])
            sdev_rnrmses.append(column_sdevs.loc['range_normalized_rmse'])
            sdev_qnrmses.append(column_sdevs.loc['interquartile_normalized_rmse'])

        # make plot and save
        fig, ax = plt.subplots(3, 1, figsize=(18, 18), sharex=True)
        ax[0].scatter(idxs, mean_r2s)
        ax[0].errorbar(idxs, mean_r2s, yerr=sdev_r2s)
        ax[0].set_ylabel('R2')
    
        ax[1].scatter(idxs, mean_rnrmses)
        ax[1].errorbar(idxs, mean_rnrmses, yerr=sdev_rnrmses)
        ax[1].set_ylabel('RN-RMSE')
        
        ax[2].scatter(idxs, mean_qnrmses)
        ax[2].errorbar(idxs, mean_qnrmses, yerr=sdev_qnrmses)
        ax[2].set_ylabel('IQRN-RMSE')
    
        for i in range(len(ax)):
            ax[i].grid(True, axis='y', color='#333333')
            ax[i].axvspan(-0.5, 5.5, color='#D3D3D3', alpha=0.3)
            ax[i].axvspan(11.5, 17.5, color='#D3D3D3', alpha=0.3)
            ax[i].axvspan(23.5, 29.5, color='#D3D3D3', alpha=0.3)
        
        ax[2].set_xticks(idxs)
        ax[2].set_xticklabels(treatments)
        ax[2].tick_params(axis='x', labelrotation=90)
    
        fig.suptitle(trait, y=0.99, fontsize=18)
        plt.tight_layout()
        plt.savefig(PATHS['post']/trait/f'{trait}_metrics.jpg')
        plt.close()

        # make DataFrame and save
        df = pd.DataFrame({'trait': [trait]*len(treatments),
                           'treatment': treatments,
                           'r2': mean_r2s,
                           'rn-rmse': mean_rnrmses,
                           'iqrn-rmse': mean_qnrmses,
                           'rmse': mean_rmses,
                           'r2-sdev': sdev_r2s,
                           'rn-rmse-sdev': sdev_rnrmses,
                           'iqrn-rmse-sdev': sdev_qnrmses,
                           'rmse-sdev': sdev_rmses})
        df.to_csv(PATHS['post']/trait/f'{trait}_metrics.csv', index=False) 
        trait_metrics_dfs.append(df)

    print('--- Making ALL_metrics.csv')
    all_df = pd.concat(trait_metrics_dfs, axis=0, ignore_index=True)
    all_df.to_csv(PATHS['post']/'ALL/ALL_metrics.csv', index=False)


def make_truepreds() -> None:
    PATHS = get_paths()
    TRAITS = get_traits()
    trait_deploys = get_deploy_by_trait()

    print('True-vs-Predictions ...')
    for trait in TRAITS.values():
        print(f'--- Making {trait}_truepreds.jpg')
        trait_deploy = get_6x6_trait_deploy(trait_deploys[trait])
        n_rows, n_cols = len(trait_deploy), len(trait_deploy[0][1])
        
        fig, ax = plt.subplots(6, 6, figsize=(30, 30), sharex=True, sharey=True)
        for (row, (treat, col_dirs)) in enumerate(trait_deploy):
            for (col, col_dir) in enumerate(col_dirs):
                preds_df = pd.read_csv(col_dir/'preds.csv')
                
                colors_df = pd.DataFrame({'sample_id': preds_df['sample_id']})
                colors_df['color'] = start_colors[col]

                metrics_df = pd.read_csv(col_dir/'metrics.csv')
                r2 = np.mean(metrics_df['r2'].values)
                rnrmse = np.mean(metrics_df['range_normalized_rmse'].values)
                stats = (f'R2 = {r2:.2f}\n'
                         f'RNRMSE = {rnrmse:.1f}')

                ax[row, col] = H.plot_pred_vs_true(ax[row, col],
                                                   pred_df=preds_df,
                                                   color_df=colors_df)
                ax[row, col].text(0.07, 
                                  0.85, 
                                  stats, 
                                  fontsize=9,
                                  transform=ax[row, col].transAxes,
                                  horizontalalignment='left')
        for r in range(6):
            ax[r, 0].set_ylabel('Predicted')
            ax[r, 0].set_title(trait_deploy[r][0])
        for c in range(6):
            ax[-1, c].set_xlabel('True')
        fig.suptitle(trait, y=0.99, fontsize=18)
        plt.tight_layout()
        plt.savefig(PATHS['post']/trait/f'{trait}_truepreds.jpg')
        plt.close()


def make_stdcoeffs() -> None:
    PATHS = get_paths()
    TRAITS = get_traits()
    trait_models = get_model_by_trait()

    print('Standard coefficients ...')
    for trait in TRAITS.values():
        print(f'--- Making {trait}_stdcoeffs.jpg')
        trait_model = get_6x6_trait_model(trait_models[trait])
        
        fig, ax = plt.subplots(6, 1, figsize=(24, 16), sharex=True)
        for (row, (treat, col_dirs)) in enumerate(trait_model):
            for (col, col_dir) in enumerate(col_dirs):
                model = H.plsr_load_model(col_dir/'model.npz')
                ax[row] = H.plot_stat_vs_wavelength(ax=ax[row],
                                                    waves=model.get(key='wavelengths'),
                                                    wave_ranges=model.get(key='kept_wave_ranges'),
                                                    samples=model.get(key='std_coefficients'),
                                                    color=start_colors[col],
                                                    show_sdev=False)
                
        for r in range(6):
            ax[r].set_ylabel('Standard coefficient')
            ax[r].set_title(trait_model[r][0])
        ax[-1].set_xlabel('Wavelengths')
        fig.suptitle(trait, y=0.99, fontsize=18)
        plt.tight_layout()
        plt.savefig(PATHS['post']/trait/f'{trait}_stdcoeffs.jpg')
        plt.close()


def make_vips() -> None:
    PATHS = get_paths()
    TRAITS = get_traits()
    trait_models = get_model_by_trait()

    print('VIPs ...')
    for trait in TRAITS.values():
        print(f'--- Making {trait}_vips.jpg')
        trait_model = get_6x6_trait_model(trait_models[trait])
        
        fig, ax = plt.subplots(6, 1, figsize=(24, 16), sharex=True)
        for (row, (treat, col_dirs)) in enumerate(trait_model):
            for (col, col_dir) in enumerate(col_dirs):
                model = H.plsr_load_model(col_dir/'model.npz')
                ax[row] = H.plot_stat_vs_wavelength(ax=ax[row],
                                                    waves=model.get(key='wavelengths'),
                                                    wave_ranges=model.get(key='kept_wave_ranges'),
                                                    samples=model.get(key='vips'),
                                                    color=start_colors[col],
                                                    show_sdev=False)
                
        for r in range(6):
            ax[r].set_ylabel('VIP')
            ax[r].set_title(trait_model[r][0])
        ax[-1].set_xlabel('Wavelengths')
        fig.suptitle(trait, y=0.99, fontsize=18)
        plt.tight_layout()
        plt.savefig(PATHS['post']/trait/f'{trait}_vips.jpg')
        plt.close()


def make_edas() -> None:
    PATHS = get_paths()
    TRAITS = get_traits()
    trait_edas = get_eda_by_trait()

    print('EDAs ...')
    for trait in TRAITS.values():
        print(f'--- Making {trait}_eda.jpg')
        trait_eda = trait_edas[trait]
        fig, ax = plt.subplots(6, 6, figsize=(24, 24))
        
        for (col, (treat, _)) in enumerate(trait_eda):
            ax[0, col].set_title(treat)
    
        for (col, (treat, eda_dir)) in enumerate(trait_eda):
            # univar R2 in first row
            data = np.load(eda_dir/'uni-r2.npz')
            metrics, wavelengths = data['metrics'], data['wavelengths']
            ax[0, col] = H.plot_uni_metric_vs_wavelength(ax=ax[0, col],
                                                         metrics=metrics.flatten(),
                                                         min_max=(0, 1),
                                                         waves=wavelengths.flatten())
    
            # univar Pearson in second row
            data = np.load(eda_dir/'uni-pearson-correlation.npz')
            metrics, wavelengths = data['metrics'], data['wavelengths']
            ax[1, col] = H.plot_uni_metric_vs_wavelength(ax=ax[1, col],
                                                         metrics=metrics.flatten(),
                                                         min_max=(-1, 1),
                                                         waves=wavelengths.flatten())
            ax[1, col].axhline(y=0, color='black')
    
            # ndi R2 heatmap, histogram in third row, fourth row respectively
            data = np.load(eda_dir/'ndi-r2.npz')
            metrics, wavelengths = data['metrics'], data['wavelengths']
            cmap = plt.cm.viridis
            cmap.set_under('white')
            norm = Normalize(vmin=0, vmax=1.0)
            ax[2, col] = H.plot_ndi_as_heatmap(ax=ax[2, col],
                                               metrics=metrics,
                                               cmap=cmap,
                                               norm=norm,
                                               waves=wavelengths.flatten(),
                                               colorbar=(col == 0))
            ax[3, col] = H.plot_ndi_as_histogram(ax=ax[3, col],
                                                 metrics=metrics,
                                                 min_max=(0, 1))
            ax[3, col].tick_params(axis='x', labelrotation=90)
    
    
            # ndi Pearson heatmap in fifth row, sixth row respectively
            data = np.load(eda_dir/'ndi-pearson-correlation.npz')
            metrics, wavelengths = data['metrics'], data['wavelengths']
            cmap = plt.cm.viridis
            cmap.set_under('white')
            norm = Normalize(vmin=-1.0, vmax=1.0)
            ax[4, col] = H.plot_ndi_as_heatmap(ax=ax[4, col],
                                               metrics=metrics,
                                               cmap=cmap,
                                               norm=norm,
                                               waves=wavelengths.flatten(),
                                               colorbar=(col==0))
            ax[5, col] = H.plot_ndi_as_histogram(ax=ax[5, col],
                                                 metrics=metrics,
                                                 min_max=(-1, 1))
            ax[5, col].tick_params(axis='x', labelrotation=90)
    
            
        ax[0, 0].set_ylabel('Univariate-R2')
        ax[1, 0].set_ylabel('Univariate-Pearson')
        ax[2, 0].set_ylabel('NDI-R2')
        ax[3, 0].set_ylabel('NDI-R2 histogram')
        ax[4, 0].set_ylabel('NDI-Pearson')
        ax[5, 0].set_ylabel('NDI-Pearson histogram')
    
        fig.suptitle(trait, y=0.99, fontsize=18)
        plt.tight_layout()
        plt.savefig(PATHS['post']/trait/f'{trait}_eda.jpg')
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Post')
    parser.add_argument('--metrics', action='store_true', default=False)
    parser.add_argument('--truepreds', action='store_true', default=False)
    parser.add_argument('--stdcoeffs', action='store_true', default=False)
    parser.add_argument('--vips', action='store_true', default=False)
    parser.add_argument('--edas', action='store_true', default=False)
    args = parser.parse_args().__dict__

    PATHS = get_paths()
    TRAITS = get_traits()
    for trait in TRAITS.values():
        (PATHS['post']/trait).mkdir(parents=True, exist_ok=True)
    (PATHS['post']/'ALL').mkdir(parents=True, exist_ok=True)
    
    if args['metrics']:
        make_metrics()

    if args['truepreds']:
        make_truepreds()

    if args['stdcoeffs']:
        make_stdcoeffs()

    if args['vips']:
        make_vips()

    if args['edas']:
        make_edas()
    
