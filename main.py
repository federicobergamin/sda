import wandb
import os
from typing import *

import hydra
from hydra.utils import call, instantiate
from omegaconf import OmegaConf

from sda.mcs import *
from sda.score import *
from sda.utils import *

from experiments.lorenz.utils import *


@hydra.main(config_path="config", config_name="main", version_base="1.3.2")
def main(cfg):
    os.environ["HYDRA_FULL_ERROR"] = "1"
    # print(OmegaConf.to_yaml(cfg))
    cfg_to_save = OmegaConf.to_container(cfg, resolve=True)
    run = wandb.init(project='sda-lorenz', group='local', config=cfg_to_save)
    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    save_config(cfg_to_save, runpath)

    # Network
    window = cfg.net.window
    score = make_local_score(**cfg.net)
    sde = VPSDE(score.kernel, shape=(window * 3,)).cuda()

    # Data
    trainset = load_data(PATH / 'data/train.h5', window=window)
    validset = load_data(PATH / 'data/valid.h5', window=window)

    # Training
    optimizer = instantiate(cfg.optim, params=sde.parameters())
    generator = loop(
        sde,
        trainset,
        validset,
        optimizer,
        **cfg,
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
    main()