import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
from itertools import product
from pprint import pprint
import shutil 
import tarfile

from utils import get_paths


def handle_job_targz(job_targz: Path) -> None:
    '''
    Extracts and puts contents of targz in eda/train/deploy

    `job_targz`: Path
                 File to extract.
                 If *.train.tar.gz, contents put into io/model and io.deploy.
                 If *.eda.tar.gz, contents put into io/eda.
    '''
    PATHS = get_paths()

    temp_dir = PATHS['fromchtc']/'job_temp'
    temp_dir.mkdir(parents=True, exist_ok=True)

    if '.train' in job_targz.stem:
        # extract job targz into temp_dir
        with tarfile.open(job_targz, "r:gz") as tar:
            tar.extractall(path=temp_dir)

        # collect nested targzs in temp_dir
        nested_targzs = [f for f in temp_dir.rglob('*.tar.gz')]
        # pprint(nested_targzs)

        # extract the nested targzs
        PATHS['model'].mkdir(parents=True, exist_ok=True)  
        PATHS['deploy'].mkdir(parents=True, exist_ok=True)  
        deploy_keys = ['__TEST__', 
                       '__FULL__', 
                       '__TRAIN__',
                       '__VALID__',
                       '__CALIB__']
        for nested_targz in nested_targzs:
            is_deploy = False
            for key in deploy_keys:
                if key in nested_targz.stem:
                    is_deploy = True
            # print(nested_targz.stem, is_deploy)

            if is_deploy:
                with tarfile.open(nested_targz, "r:gz") as tar:
                    tar.extractall(path=PATHS['deploy'])
            else:
                with tarfile.open(nested_targz, "r:gz") as tar:
                    tar.extractall(path=PATHS['model'])
    elif '.eda' in job_targz.stem:
        # extract job targz into temp_dir
        with tarfile.open(job_targz, "r:gz") as tar:
            tar.extractall(path=temp_dir)

        # collect nested targzs in temp_dir
        nested_targzs = [f for f in temp_dir.rglob('*.tar.gz')]
        # pprint(nested_targzs)

        # extract nested targzs into PATHS['eda']
        PATHS['eda'].mkdir(parents=True, exist_ok=True)  
        for nested_targz in nested_targzs:
            with tarfile.open(nested_targz, "r:gz") as tar:
                tar.extractall(path=PATHS['eda'])

    shutil.rmtree(temp_dir)
            

if __name__ == '__main__':
    PATHS = get_paths()

    # create a temp directory
    temp_dir = PATHS['fromchtc']/'proj_temp'
    temp_dir.mkdir(parents=True, exist_ok=True)

    n_targzs = 0
    # collect all tar.gz files
    proj_targzs = [f for f in PATHS['fromchtc'].glob('*.tar.gz')]
    
    for proj_targz in proj_targzs:
        with tarfile.open(proj_targz, "r:gz") as tar:
            tar.extractall(path=temp_dir)
    
        job_targzs = [f for f in temp_dir.rglob('*.tar.gz')]
        job_targzs.sort()
    
        for job_targz in job_targzs:
            handle_job_targz(job_targz=job_targz)
            n_targzs += 1

    print(f'{n_targzs = }')
    shutil.rmtree(temp_dir)
