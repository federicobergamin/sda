import wandb
import os
from typing import *

import hydra
from hydra.utils import call, instantiate
from omegaconf import OmegaConf

from sda.mcs import *
from sda.score import *
from sda.utils import *
from sda.utils.loggers import LoggerCollection
from experiments.lorenz.utils import *


@hydra.main(config_path="config", config_name="main", version_base="1.3.2")
def main(cfg):
    os.environ["HYDRA_FULL_ERROR"] = "1"
    cfg_to_save = OmegaConf.to_container(cfg, resolve=True)

    loggers = [instantiate(logger_config) for logger_config in cfg.logger.values()]
    logger = LoggerCollection(loggers)
    logger.log_hyperparams(cfg_to_save)

    # Network
    window = cfg.net.window
    if cfg.score.local:
        score = make_local_score(**cfg.net)
    else:
        score = make_global_score(**cfg.net)
    sde = VPSDE(score.kernel, shape=(window * 3,)).cuda()

    # Data
    trainset = load_data(os.path.join(cfg.work_dir, 'data/train.h5'), window=window)
    validset = load_data(os.path.join(cfg.work_dir, 'data/valid.h5'), window=window)

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
        logger.log_metrics({
            'loss_train': loss_train,
            'loss_valid': loss_valid,
            'lr': lr,
        })

    # Save
    torch.save(
        score.state_dict(),
        f"{cfg.ckpt_dir} / score.pth",
    )

    # Evaluation
    # NOTE [Understand if it's correct or not]: here at evaluation time the score of our SDE is just the score.kernel
    # so we are not using the Algorithm 2 here to constuct the score neither.
    chain = make_chain()

    x = sde.sample((4096,), steps=64).cpu()
    x = x.unflatten(-1, (-1, 3))
    x = chain.postprocess(x)

    log_p = chain.log_prob(x[:, :-1], x[:, 1:]).mean()

    logger.log_metrics({'log_p': log_p})
    logger.close()


if __name__ == '__main__':
    main()