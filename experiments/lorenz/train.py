#!/usr/bin/env python

import wandb

from dawgz import job, schedule
from typing import *

from sda.mcs import *
from sda.score import *
from sda.utils import *

from utils import *


GLOBAL_CONFIG = {
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

LOCAL_CONFIG = {
    # Architecture
    'window': 5,
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


@job(array=3, cpus=2, gpus=1, ram='8GB', time='06:00:00')
def train_global(i: int):
    run = wandb.init(project='sda-lorenz', group='global', config=GLOBAL_CONFIG)
    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    save_config(GLOBAL_CONFIG, runpath)

    # Network
    score = make_global_score(**GLOBAL_CONFIG)
    sde = VPSDE(score, shape=(3, 32)).cuda()

    # Data
    trainset = load_data(PATH / 'data/train.h5').unfold(1, 32, 1).flatten(0, 1)
    validset = load_data(PATH / 'data/valid.h5').unfold(1, 32, 1).flatten(0, 1)

    # Training
    generator = loop(
        sde,
        trainset,
        validset,
        **GLOBAL_CONFIG,
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

    x = sde.sample((1024,), steps=64).cpu()
    x = x.transpose(-1, -2)
    x = chain.postprocess(x)

    log_p = chain.log_prob(x[:, :-1], x[:, 1:]).mean()

    run.log({'log_p': log_p})
    run.finish()


@job(array=3, cpus=2, gpus=1, ram='8GB', time='06:00:00')
def train_local(i: int):
    run = wandb.init(project='sda-lorenz', group='local', config=LOCAL_CONFIG)
    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    save_config(LOCAL_CONFIG, runpath)

    # Network
    window = LOCAL_CONFIG['window']
    score = make_local_score(**LOCAL_CONFIG)
    sde = VPSDE(score.kernel, shape=(window * 3,)).cuda()

    # Data
    trainset = load_data(PATH / 'data/train.h5', window=window)
    validset = load_data(PATH / 'data/valid.h5', window=window)

    # Training
    generator = loop(
        sde,
        trainset,
        validset,
        **LOCAL_CONFIG,
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
    # NOTE [Understand if it's correct or not]: here at evaluation time the score of our SDE is just the score.kernel
    # so we are not using the Algorithm 2 here to constuct the score neither.
    chain = make_chain()

    x = sde.sample((4096,), steps=64).cpu()
    x = x.unflatten(-1, (-1, 3))
    x = chain.postprocess(x)

    log_p = chain.log_prob(x[:, :-1], x[:, 1:]).mean()

    run.log({'log_p': log_p})
    run.finish()


if __name__ == '__main__':
    schedule(
        train_global,
        train_local,
        name='Training',
        backend='slurm',
        settings={'export': 'ALL'},
        env=[
            'conda activate sda',
            'export WANDB_SILENT=true',
        ],
    )
