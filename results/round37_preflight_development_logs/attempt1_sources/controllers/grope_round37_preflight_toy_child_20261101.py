import sys
from pathlib import Path
ROOT=Path('/home/liwt/GRoPE');sys.path.insert(0,str(ROOT))
from experiments import round37_component_round_real_preflight as P
import numpy as np
import pytest
def forbidden(*args,**kwargs):raise AssertionError('pure runner test may not call real reference integration')
P.REF.integrate=forbidden
original_load=np.load
def guarded(path,*args,**kwargs):
    if isinstance(path,(str,Path)) and Path(path).resolve().is_relative_to(ROOT/'results'):
        raise AssertionError('pure runner test may not decode formal saved inputs/outputs')
    return original_load(path,*args,**kwargs)
np.load=guarded
sys.exit(pytest.main(['-q','tests/test_round37_component_round_real_preflight.py']))
