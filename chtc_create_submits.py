from __future__ import annotations
from typing import List, Dict, Tuple, Union
import sys
from pathlib import Path
import argparse
from collections import defaultdict
from pprint import pprint
import string
import shutil

from utils import get_cli_args, get_paths
from project import get_model_names


if __name__ == '__main__':
    ARGS = get_cli_args()
    
    USER = ARGS['chtc_user']
    PROJECT = ARGS['chtc_project_name']
    MODELS_PER_SUBMIT = int(ARGS['chtc_n_models_per_submit'])
    NOUTERS = str(ARGS['n_outers'])
    NINNERS = str(ARGS['n_inners'])
    MINSAMPLES = int(ARGS['min_n_samples'])
    PATTERN = ARGS['pattern']
    
    PATHS = get_paths()
    chtc_dir = PATHS['chtc']
    gens_dir = chtc_dir/'gens'
    refs_dir = chtc_dir/'refs'
 
    # cleanup previous generated files. 
    if gens_dir.exists():
        shutil.rmtree(str(gens_dir))
    gens_dir.mkdir(parents=True, exist_ok=True)

    model_names = get_model_names(min_n_samples=MINSAMPLES,
                                  pattern=PATTERN)
    model_names.sort()
    
    n_models = len(model_names)
    if MODELS_PER_SUBMIT == -1:
        MODELS_PER_SUBMIT = n_models

    # create params and submit files
    n_submits = []
    for (i, k) in enumerate(range(0, n_models, MODELS_PER_SUBMIT)):
        model_names_ = model_names[k:(k + MODELS_PER_SUBMIT)]

        # params.model.i
        params_model = []
        for mn in model_names_:
            p = [USER, PROJECT, NOUTERS, NINNERS, mn]
            params_model.append(','.join(p))
        params_model_file = gens_dir/f'params.model.{i}' 
        with open(params_model_file, 'w') as writer:
            writer.write('\n'.join(params_model))

        # submit.model.i
        with open(refs_dir/'submit.model.template', 'r') as reader:
            template = reader.read()
        submit_contents = template.replace('PARAMSFILE', 
                                           params_model_file.name)
        submit_model_file = gens_dir/f'submit.model.{i}'
        with open(submit_model_file, 'w') as writer:
            writer.write(submit_contents)

        # params.packstaging.i
        params_pack = [] 
        p = [USER, PROJECT, ':'.join(model_names_)]
        params_pack.append(','.join(p))
        params_pack_file = gens_dir/f'params.packstaging.{i}' 
        with open(params_pack_file, 'w') as writer:
            writer.write('\n'.join(params_pack))

        # submit.packstaging.i
        with open(refs_dir/'submit.packstaging.template', 'r') as reader:
            template = reader.read()
        submit_contents = template.replace('PARAMSFILE', 
                                           params_pack_file.name)
        submit_pack_file = gens_dir/f'submit.packstaging.{i}'
        with open(submit_pack_file, 'w') as writer:
            writer.write(submit_contents)

        n_submits.append(len(model_names_))

    # create dag submit file
    dag_lines = []
    alphabets = list(string.ascii_uppercase)
    for i in range(len(n_submits)):
        dag_lines.append(f'JOB {alphabets[2*i]} submit.model.{i}')
        dag_lines.append(f'JOB {alphabets[2*i + 1]} submit.packstaging.{i}')
    n_dag_submits = len(dag_lines)
    for i in range(1, n_dag_submits):
        p, c = alphabets[i - 1], alphabets[i]
        dag_lines.append(f'PARENT {p} CHILD {c}')
    with open(gens_dir/'submit.dag', 'w') as writer:
        writer.write('\n'.join(dag_lines))

    # print: summary
    for (i, n) in enumerate(n_submits):
        print(f'Stage {i}: {n} models.')
    print(f'Total number of models: {n_models}.')
