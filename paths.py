from __future__ import annotations
from typing import List, Dict
from pathlib import Path

def get_paths() -> Path:
    io_dir = Path(__file__).parent/'io'
    
    return {'original': io_dir/'original',
            'compatible': io_dir/'compatible',
            'config': io_dir/'config',
            'model': io_dir/'model', 
            'ideploy': io_dir/'ideploy',
            'edeploy': io_dir/'edeploy'}
