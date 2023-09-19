'''
File to generate the kolmogorov dataset used in the experiments
and avoid using the dawgz library which is not really working in our 
slurm cluster
'''
import h5py
import numpy as np
import random

# from dawgz import job, after, ensure, schedule
from typing import *

from sda.mcs import *
from sda.utils import *

from utils import *
from pathlib import Path

def main():
    # let's start by creating the correct directory
    # NOTE: since I am reaching the files quota, I will save everything
    # in scratch
    PATH = Path('/scratch/fedbe/') / 'sda/kolmogorov'
    (PATH / 'data').mkdir(parents=True, exist_ok=True)
    print(PATH)

    # then we have to create 1024 trajectories of lenght 128
    for i in range(3):
        # we create a KolmogorovFlow object
        chain = make_chain()
        random.seed(i)

        x = chain.prior()
        x = chain.trajectory(x, length=128)
        x = x[64:]

        np.save(PATH + f'data/x_{i:06d}.npy', x)

    
    # and once we have all the different trajectories created,
    # we can aggregate them all and create the dataset and split
    # it into train, valid and test
    # files = list(PATH.glob('data/x_*.npy'))
    # length = len(files)

    # i = int(0.8 * length)
    # j = int(0.9 * length)

    # splits = {
    #     'train': files[:i],
    #     'valid': files[i:j],
    #     'test': files[j:],
    # }

    # for name, files in splits.items():
    #     with h5py.File(PATH / f'data/{name}.h5', mode='w') as f:
    #         f.create_dataset(
    #             'x',
    #             shape=(len(files), 64, 2, 64, 64),
    #             dtype=np.float32,
    #         )

    #         for i, x in enumerate(map(np.load, files)):
    #             x = torch.from_numpy(x)
    #             f['x'][i] = KolmogorovFlow.coarsen(x, 4)


if __name__ == "__main__":  
    main()