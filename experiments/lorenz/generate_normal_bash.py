'''
File used to generate the data for the Lorentz experiment.
Was not able to run the file with all the dawgz tags even on
a slurm based cluster.
'''

from typing import *

from sda.mcs import *
from sda.utils import *

from utils import *
# from pathlib import Path

def main():
    # if 'SCRATCH' in os.environ:
    #     SCRATCH = os.environ['SCRATCH']
    #     PATH = Path(SCRATCH) / 'sda/lorenz'
    # else:
    #     PATH = Path('.')

    # PATH.mkdir(parents=True, exist_ok=True)
    
    # we start by creating the folder containing the data
    (PATH / 'data').mkdir(parents=True, exist_ok=True)

    # thenI have to simulate the chain
    chain = make_chain()

    x = chain.prior((1024,))
    x = chain.trajectory(x, length=1024, last=True)
    x = chain.trajectory(x, length=1024)
    x = chain.preprocess(x)
    x = x.transpose(0, 1)

    i = int(0.8 * len(x))
    j = int(0.9 * len(x))

    splits = {
        'train': x[:i],
        'valid': x[i:j],
        'test': x[j:],
    }

    for name, x in splits.items():
        save_data(x, PATH / f'data/{name}.h5')

if __name__ == "__main__":  
    main()