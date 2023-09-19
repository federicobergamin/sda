'''
Modifying the train file to run it by using a bash script because the dawgz
library does not work with my cluster.

TODO: [x] check that everything runs
      [x] data is local, but save everything on scratch
'''

import wandb

from typing import *

from sda.mcs import *
from sda.score import *
from sda.utils import *

from utils import *
import argparse
from pathlib import Path

def main(args):
    DATA_PATH = '../experiments/lorenz/data/'
    SCRATCH_PATH = Path('/scratch/fedbe/sda/experiments/lorenz')
    # os.makedirs(SCRATCH_PATH / 'runs/', exist_ok=True)


    if args.experiment == 'global':
        CONFIG = {
            # Architecture
            'embedding': 32,
            'hidden_channels': (64,),
            'hidden_blocks': (3,),
            'activation': 'SiLU',
            # Training
            'epochs': 1024,
            'epoch_size': 16384,
            'batch_size': 64,
            'optimizer': 'AdamW',
            'learning_rate': 1e-3,
            'weight_decay': 1e-3,
            'scheduler': 'linear',
        }

        # set-up wandb
        run = wandb.init(project='sda-lorenz', group='global', config=CONFIG)
        runpath = SCRATCH_PATH / f'runs/{run.name}_{run.id}'
        runpath.mkdir(parents=True, exist_ok=True)

        save_config(CONFIG, runpath)

        # Network
        score = make_global_score(**CONFIG)
        sde = VPSDE(score, shape=(3, 32)).cuda()

        # Data
        trainset = load_data(DATA_PATH + 'train.h5').unfold(1, 32, 1).flatten(0, 1)
        validset = load_data(DATA_PATH + 'valid.h5').unfold(1, 32, 1).flatten(0, 1)
        
    elif args.experiment == 'local':
        CONFIG = {
            # Architecture
            'window': args.window,
            'embedding': 32,
            'width': 256,
            'depth': 5,
            'activation': 'SiLU',
            # Training
            'epochs': 1024,
            'epoch_size': 65536,
            'batch_size': 256,
            'optimizer': 'AdamW',
            'learning_rate': 1e-3,
            'weight_decay': 1e-3,
            'scheduler': 'linear',
        }

        # Wandb set-up
        run = wandb.init(project='sda-lorenz', group='local', config=CONFIG)
        runpath = SCRATCH_PATH / f'runs/{run.name}_{run.id}'
        runpath.mkdir(parents=True, exist_ok=True)

        save_config(CONFIG, runpath)

        # Network
        window = CONFIG['window']
        # window = args.window
        score = make_local_score(**CONFIG)
        sde = VPSDE(score.kernel, shape=(window * 3,)).cuda()

        # Data
        trainset = load_data(DATA_PATH + 'train.h5', window=window)
        validset = load_data(DATA_PATH + 'valid.h5', window=window)

    else:
        raise NotImplementedError('Argument experiment can be only in [global, local]')
    

    # now we have everything to start the training loop
    # Training
    generator = loop(
        sde,
        trainset,
        validset,
        **CONFIG,
        device='cuda',
    )

    for loss_train, loss_valid, lr in generator:
        run.log({
            'loss_train': loss_train,
            'loss_valid': loss_valid,
            'lr': lr,
        })

    # Save
    torch.save(
        score.state_dict(),
        runpath / f'state.pth',
    )

    # Evaluation
    chain = make_chain()

    if args.experiment == 'global':
        x = sde.sample((1024,), steps=64).cpu()
        x = x.transpose(-1, -2)
    else:
        x = sde.sample((4096,), steps=64).cpu()
        x = x.unflatten(-1, (-1, 3))

    x = chain.postprocess(x)

    log_p = chain.log_prob(x[:, :-1], x[:, 1:]).mean()

    run.log({'log_p': log_p})
    run.finish()


if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description='Train Lorenz Experiments')
    # parser.add_argument('--seed', '-seed', type=int, default=0, help='seed for randomness generator')
    parser.add_argument('--seed', '-s', type=int, default=77, help='seed')
    parser.add_argument('--experiment', '-exp', type=str, default='local', help='local: k <= 4, global: k>4')
    parser.add_argument('--window', '-w', type=int, default=5, help='seed')
    args = parser.parse_args()
    main(args)
