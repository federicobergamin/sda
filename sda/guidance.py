from abc import ABC, abstractmethod

from torch import Tensor, nn
from torch.func import grad_and_value, grad, vmap

from .likelihood import Likelihood
from .sde import VPSDE


class GuidedScore(ABC, nn.Module):
    def __init__(self, sde: VPSDE, likelihood: Likelihood):
        self.likelihood = likelihood
        self.sde = sde

    def get_sde(self, shape) -> VPSDE:
        return self.sde.__class__(self, shape=shape)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass


class DPS(GuidedScore):
    def __init__(self, sde: VPSDE, likelihood: Likelihood, zeta: float = 1.0):
        super().__init__(sde, likelihood)
        self.zeta = zeta

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)
        x = x.detach().requires_grad_(True)

        def log_prob(x):
            eps = self.sde.eps(x, t)
            x_ = (x - sigma * eps) / mu
            err = self.likelihood.err(x_).square().sum()
            return err, eps

        guidance, err, eps = grad_and_value(log_prob, has_aux=True)(x)
        guidance = guidance * self.zeta / err.sqrt()
        return eps - sigma * guidance


class Reconstruction(GuidedScore):
    def __init__(
            self,
            sde: VPSDE,
            likelihood: Likelihood,
            gamma: float = 1.0
            ):
        super().__init__(sde, likelihood)
        self.gamma = gamma

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)
        x = x.detach().requires_grad_(True)

        def log_prob(x):
            eps = self.sde.eps(x, t)
            x_ = (x - sigma * eps) / mu
            var_x0_xt = self.gamma * (sigma / mu) ** 2
            err = self.likelihood.log_prob(x_, var_x0_xt)
            return err, eps

        guidance, eps = grad(log_prob, has_aux=True)(x)
        return eps - sigma * guidance