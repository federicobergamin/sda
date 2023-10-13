r"""Score modules"""

import math
import torch
import torch.nn as nn

from torch import Size, Tensor
from tqdm import tqdm
from typing import *
from zuko.utils import broadcast


class VPSDE(nn.Module):
    r"""Creates a noise scheduler for the variance preserving (VP) SDE.

    .. math::
        \mu(t) & = \alpha(t) \\
        \sigma(t)^2 & = 1 - \alpha(t)^2 + \eta^2

    Arguments:
        eps: A noise estimator :math:`\epsilon_\phi(x, t)`.
        shape: The event shape.
        alpha: The choice of :math:`\alpha(t)`.
        eta: A numerical stability term.
    """

    def __init__(
        self,
        eps: nn.Module,
        shape: Size = (),
        alpha: str = 'cos',
        eta: float = 1e-3,
    ):
        super().__init__()

        self.eps = eps
        self.shape = shape
        self.dims = tuple(range(-len(shape), 0))
        self.eta = eta

        if alpha == 'lin':
            self.alpha = lambda t: 1 - (1 - eta) * t
        elif alpha == 'cos':
            self.alpha = lambda t: torch.cos(math.acos(math.sqrt(eta)) * t) ** 2
        elif alpha == 'exp':
            self.alpha = lambda t: torch.exp(math.log(eta) * t**2)
        else:
            raise ValueError()

        self.register_buffer('device', torch.empty(()))

    def mu(self, t: Tensor) -> Tensor:
        return self.alpha(t)

    def sigma(self, t: Tensor) -> Tensor:
        return (1 - self.alpha(t) ** 2 + self.eta ** 2).sqrt()

    def forward(self, x: Tensor, t: Tensor, train: bool = False) -> Tensor:
        r"""Samples from the perturbation kernel :math:`p(x(t) | x)`."""

        t = t.reshape(t.shape + (1,) * len(self.shape))

        eps = torch.randn_like(x)
        x = self.mu(t) * x + self.sigma(t) * eps

        if train:
            return x, eps
        else:
            return x

    def sample(
        self,
        shape: Size = (),
        steps: int = 64,
        corrections: int = 0,
        tau: float = 1.0,
    ) -> Tensor:
        r"""Samples from :math:`p(x(0))`.

        Arguments:
            shape: The batch shape.
            steps: The number of discrete time steps.
            corrections: The number of Langevin corrections per time steps.
            tau: The amplitude of Langevin steps.
        """

        x = torch.randn(shape + self.shape).to(self.device)
        x = x.reshape(-1, *self.shape)

        time = torch.linspace(1, 0, steps + 1).to(self.device)
        dt = 1 / steps

        with torch.no_grad():
            for t in tqdm(time[:-1]):
                # Predictor
                r = self.mu(t - dt) / self.mu(t)
                x = r * x + (self.sigma(t - dt) - r * self.sigma(t)) * self.eps(x, t)

                # Corrector
                for _ in range(corrections):
                    eps = torch.randn_like(x)
                    # NOTE: since we are parametrizing eps(x(t),t) = - sigma(t) * score(x(t),t)
                    # if we need to use score(x(t),t) we have to always compute
                    # score(x(t),t) = - eps(x(t),t) / sigma(t)
                    s = -self.eps(x, t - dt) / self.sigma(t - dt)
                    delta = tau / s.square().mean(dim=self.dims, keepdim=True)

                    x = x + delta * s + torch.sqrt(2 * delta) * eps

        return x.reshape(shape + self.shape)

    def loss(self, x: Tensor) -> Tensor:
        r"""Returns the denoising loss."""

        t = torch.rand(x.shape[0], dtype=x.dtype, device=x.device)
        x, eps = self.forward(x, t, train=True)

        return (self.eps(x, t) - eps).square().mean()


class SubVPSDE(VPSDE):
    r"""Creates a noise scheduler for the sub-variance preserving (sub-VP) SDE.

    .. math::
        \mu(t) & = \alpha(t) \\
        \sigma(t)^2 & = (1 - \alpha(t)^2 + \eta)^2
    """

    def sigma(self, t: Tensor) -> Tensor:
        return 1 - self.alpha(t) ** 2 + self.eta


class SubSubVPSDE(VPSDE):
    r"""Creates a noise scheduler for the sub-sub-VP SDE.

    .. math::
        \mu(t) & = \alpha(t) \\
        \sigma(t)^2 & = (1 - \alpha(t) + \eta)^2
    """

    def sigma(self, t: Tensor) -> Tensor:
        return 1 - self.alpha(t) + self.eta
