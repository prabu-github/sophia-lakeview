import sys
from pathlib import Path
import numpy as np
import pandas as pd
from pandas import DataFrame
from pprint import pprint
import time

hytraits_path = Path('../hytraits').resolve()
if str(hytraits_path) not in sys.path:
    sys.path.append(str(hytraits_path))
import hytraits as H

np.set_printoptions(precision=8)

if __name__ == '__main__':
    param_json = Path(sys.argv[1]).resolve()
    print(f'Training: {param_json.name} ...')
    TRAINER = H.TraitTrainer()
    TRAINER(param_json=param_json) 
