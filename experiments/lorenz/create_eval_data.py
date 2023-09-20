'''
File where I try to rewrite the code that is using the dawgz
library in the usual way that support bash scripts.
'''

from h5py import *
from typing import *

from sda.mcs import *
from sda.score import *
from sda.utils import *

from utils import *
import argparse
from pathlib import Path

def main():
    # DATA_PATH = Path('data')

    # to study impact of k and number of corrected samples 
    # we consider two simple observation processes N (y | a_{1:L:8},0.05^2 I) and N(y | a _{1:L}, 0.25^2 I)
    x = load_data(PATH / 'data/test.h5')[:, :65]
    print(x.shape)
    y_lo = torch.normal(x[:, ::8, :1], 0.05)
    y_hi = torch.normal(x[:, :, :1], 0.25)
    
    print('creating the folder results')
    (PATH / 'results').mkdir(parents=True, exist_ok=True)

    print('Saving the data')
    with h5py.File(PATH / 'results/obs.h5', mode='w') as f:
        f.create_dataset('lo', data=y_lo)
        f.create_dataset('hi', data=y_hi)
    
if __name__ == "__main__":  
    # parser = argparse.ArgumentParser(description='Train Lorenz Experiments')
    # parser.add_argument('--seed', '-seed', type=int, default=0, help='seed for randomness generator')
    # parser.add_argument('--seed', '-s', type=int, default=77, help='seed')
    # parser.add_argument('--experiment', '-exp', type=str, default='local', help='local: k <= 4, global: k>4')
    # parser.add_argument('--window', '-w', type=int, default=5, help='seed')
    # args = parser.parse_args()
    main()
