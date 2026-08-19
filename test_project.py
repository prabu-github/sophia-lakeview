from pathlib import Path
import json
import argparse
from pprint import pprint

from paths import PATHS
from project import (get_dataset,
                      get_splits,
                      get_xtransforms,
                      get_ytransforms,
                      get_modelnames,
                      get_data,
                      fit_model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Tests for project.py.')

    parser.add_argument('--dskey',
                        choices=['sophia260424'],
                        action='store',
                        default='sophia260424')
    parser.add_argument('--get_xtransforms',
                        action='store_true',
                        default=False)
    parser.add_argument('--get_ytransforms',
                        action='store_true',
                        default=False)
    parser.add_argument('--get_modelnames',
                        action='store_true',
                        default=False)
    parser.add_argument('--pattern',
                        action='store',
                        nargs='*',
                        default=['*'])
    parser.add_argument('--get_data',
                        action='store_true',
                        default=False)
    parser.add_argument('--modelname',
                        action='store',
                        default='dplsr__gerritall-phenolics__all__id')
    parser.add_argument('--extension',
                        action='store',
                        default='parquet',
                        choices=['csv', 
                                 'parquet'])

    args = parser.parse_args()
    pprint(args)

    if args.get_xtransforms:
        xts = get_xtransforms(ds_key=args.dskey)
        for k in xts:
            print(k)
            for t in xts[k]:
                print(t.get_name_params())

    if args.get_ytransforms:
        yts = get_ytransforms(ds_key=args.dskey)
        for k in yts:
            print(k)
            for t in yts[k]:
                print(t.get_name_params())

    if args.get_modelnames:
        modelnames = get_modelnames(comp_dir=PATHS['compdata'],
                                    patterns=args.pattern)
        pprint(f'{len(modelnames) = }')
        pprint(modelnames)

    if args.get_data:
        dataset, splits = get_data(model_name=args.modelname,
                                   comp_dir=PATHS['compdata'],
                                   seed=42)
        print(f'{args.modelname = }')
        print(f'{len(dataset) = }')
        print(f'{splits.n_outers = }')
        print(f'{splits.n_inners = }')
        print(f'{dataset.wave_ranges = }')
        d = dataset[0]
        print(f'{d["wave_ranges"] = }')
    