from __future__ import annotations
from typing import List, Dict, Tuple, Union
from pathlib import Path
import argparse
import tarfile
from pprint import pprint
import shutil
import time
from utils import get_cli_args


if __name__ == '__main__':
    ARGS = get_cli_args()

    USER = ARGS['chtc_user']
    PROJECT = ARGS['chtc_project_name']
    MODELSLIST = ARGS['chtc_models_list']

    # models 
    models = sorted(list(set(MODELSLIST.split(':'))))
    
    # staging directory
    staging = Path(f'/staging/{USER}/{PROJECT}')
    
    # collect tar.gzs
    targzs = []
    for f in staging.glob('*.tar.gz'):
        if '_packed' not in f.name:
            for m in models:
                if m == f.name[:len(m)]:
                    targzs.append(f)
    targzs.sort()
    print(f'{len(targzs)} to be packed.')
    
    # packed tar.gz
    dest = f'{models[0]}__{models[-1]}__packed.tar.gz'
    with tarfile.open(staging/dest, 'w:gz') as targzer:
        for (i, f) in enumerate(targzs):
            targzer.add(f, arcname=f.name)
            print(f'Added {f.name} ({i})')
    print(f'Packed: {dest}')

    time.sleep(15)

    # remove added
    for (i, f) in enumerate(targzs):
        if f.exists():
            f.unlink()
            print(f'Deleted: {f.name} ({i})')
