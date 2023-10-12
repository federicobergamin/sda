'''
File created to use bash scripts to train the model on the Kolmogorov dataset. The dawgz
library is not well suited for our cluster.

TODO: [] check that everything run
      [] pepare bash scripts to train three models
'''

import wandb

from typing import *

from sda.mcs import *
from sda.score import *
from sda.utils import *

from utils import *
import argparse

def main(args):
    DATA_PATH = Path('./scratch/cdd/sda/kolmogorov')
    SCRATCH_PATH = Path('./scratch/cdd/sda/experiments/kolmogorov')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    CONFIG = {
        # Architecture
        'window': 5,
        'embedding': 64,
        'hidden_channels': (96, 192, 384),
        'hidden_blocks': (3, 3, 3),
        'kernel_size': 3,
        'activation': 'SiLU',
        # Training
        'epochs': 1024,
        'batch_size': 32,
        'optimizer': 'AdamW',
        'learning_rate': 2e-4,
        'weight_decay': 1e-3,
        'scheduler': 'linear',
    }

    # initialize Wandb 
    run = wandb.init(project='sda-kolmogorov', config=CONFIG)
    runpath = SCRATCH_PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    save_config(CONFIG, runpath)

    # Network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    window = CONFIG['window']
    score = make_score(**CONFIG)
    sde = VPSDE(score.kernel, shape=(window * 2, 64, 64)).to(device)

    # Data
    trainset = load_data(DATA_PATH / 'data/train.h5', window=window)
    validset = load_data(DATA_PATH / 'data/valid.h5', window=window)
    
    # Training
    generator = loop(
        sde,
        trainset,
        validset,
        device=device,
        **CONFIG,
    )

    # logging
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
    x = sde.sample((2,), steps=64).cpu()
    x = x.unflatten(1, (-1, 2))
    w = KolmogorovFlow.vorticity(x)

    run.log({'samples': wandb.Image(draw(w))})
    run.finish()

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description='Train Kolmogorov Experiments')
    # parser.add_argument('--seed', '-seed', type=int, default=0, help='seed for randomness generator')
    parser.add_argument('--seed', '-s', type=int, default=77, help='seed')
    args = parser.parse_args()
    main(args)