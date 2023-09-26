import wandb
import os
from typing import *

import hydra
from hydra.utils import call, instantiate
from omegaconf import OmegaConf

from sda.mcs import *
from sda.score import *
from sda.loop import loop
from sda.utils import *
from sda.utils.loggers import LoggerCollection


class ScoreNet(nn.Module):
    r"""Creates a simple score network made of residual blocks.

    Arguments:
        features: The number of features.
        embedding: The number of time embedding features.
    """

    # def __init__(self, features: int, embedding: int = 16, **kwargs):
    def __init__(self, net, features: int, embedding: int = 16):
        super().__init__()

        self.embedding = TimeEmbedding(embedding)
        # self.network = ResMLP(features + embedding, features, **kwargs)
        self.network = net(features + embedding, features)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t = self.embedding(t)
        x, t = broadcast(x, t, ignore=1)
        x = torch.cat((x, t), dim=-1)

        return self.network(x)


class ScoreUNet(nn.Module):
    r"""Creates a U-Net score network.

    Arguments:
        channels: The number of channels.
        embedding: The number of time embedding features.
    """

    def __init__(self, net, features: int, embedding: int = 64, **kwargs):
        super().__init__()

        self.embedding = TimeEmbedding(embedding)
        # self.network = UNet(channels, channels, embedding, **kwargs)
        self.network = net(features, features)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        dims = self.network.spatial + 1

        y = x.reshape(-1, *x.shape[-dims:])
        t = t.reshape(-1)
        t = self.embedding(t)

        return self.network(y, t).reshape(x.shape)


class SpecialScoreUNet(ScoreUNet):
    r"""Creates a score U-Net with a forcing channel."""

    def __init__(
        self,
        net, 
        features: int,
        embedding: int,
        size: int = 64,
        **kwargs,
    ):
        super().__init__(net, features + 1, embedding, **kwargs)

        domain = 2 * torch.pi / size * (torch.arange(size) + 1 / 2)
        forcing = torch.sin(4 * domain).expand(1, size, size).clone()

        self.register_buffer('forcing', forcing)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x, f = broadcast(x, self.forcing, ignore=3)
        x = torch.cat((x, f), dim=-3)

        return super().forward(x, t)[..., :-1, :, :]


class MCScoreNet(nn.Module):
    r"""Creates a score network for a Markov chain.

    Arguments:
        features: The number of features.
        order: The order of the Markov chain.
    """

    # def __init__(self, features: int, order: int = 1, **kwargs):
    def __init__(self, kernel, order: int):
        super().__init__()

        self.order = order
        self.kernel = kernel
        # self.kernel = build(features * (2 * order + 1), **kwargs)

    def forward( self, x: Tensor, t: Tensor) -> Tensor:
        x = self.unfold(x, self.order)
        s = self.kernel(x, t)
        s = self.fold(s, self.order)
        return s


def make_score(score, net, window, spatial):
    # Partially create neural network
    net = instantiate(net, _partial_=True)

    # Create score wrapper to combine with time context
    score = instantiate(score, net=net, features=spatial * window)

    # Construct full score network from markov blanket scores
    return MCScoreNet(
        kernel=score,
        order=window // 2,
    )


@hydra.main(config_path="config", config_name="main", version_base="1.3.2")
def main(cfg):
    os.environ["HYDRA_FULL_ERROR"] = "1"
    cfg_to_save = OmegaConf.to_container(cfg, resolve=True)

    loggers = [instantiate(logger_config) for logger_config in cfg.logger.values()]
    logger = LoggerCollection(loggers)
    logger.log_hyperparams(cfg_to_save)

    window = cfg.window

    # Data
    trainset = load_data(os.path.join(cfg.data.path, 'train.h5'), window=window)
    validset = load_data(os.path.join(cfg.data.path, 'valid.h5'), window=window)

    # Network
    score = make_score(cfg.score, cfg.net, window, cfg.data.spatial)
    shape = (window * cfg.data.spatial, *cfg.data.grid_size)
    sde = VPSDE(score.kernel, shape=shape).cuda()

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
    if cfg.name == 'lorenz':
        # NOTE [Understand if it's correct or not]: here at evaluation time the score of our SDE is just the score.kernel
        # so we are not using the Algorithm 2 here to constuct the score neither.
        from sda.expriments.lorenz.utils import make_chain
        chain = make_chain()

        x = sde.sample((4096,), steps=64).cpu()
        x = x.unflatten(-1, (-1, 3))
        # x = x.transpose(-1, -2) #NOTE: global
        x = chain.postprocess(x)

        log_p = chain.log_prob(x[:, :-1], x[:, 1:]).mean()
        logger.log_metrics({'log_p': log_p})

    elif cfg.name == 'kolmogorov':
        from sda.experiments.kolmogorov.utils import draw
        x = sde.sample((2,), steps=64).cpu()
        x = x.unflatten(1, (-1, 2))
        w = KolmogorovFlow.vorticity(x)
        # run.log({'samples': wandb.Image(draw(w))})
        logger.log_image('vorticity', draw(w))
    else:
        raise ValueError('cfg.name should be either lorenz or kolmogorov but got {}'.format(cfg.name))

    logger.close()


if __name__ == '__main__':
    main()