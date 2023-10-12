from abc import ABC, abstractmethod
from typing import Callable

from torch import Tensor, nn
from torch.func import grad_and_value, grad, vmap
from torch.distributions import Normal

from sda.score import VPSDE


# class ForwardOperator(ABC):
#     def __init__(self):
#         pass

#     @abstractmethod
#     def __call__(self, x: Tensor) -> Tensor:
#         pass


######################################################
################## Likelihood #####################
######################################################

class Likelihood(ABC, nn.Module):
    def __init__(self, y: Tensor, A: Callable[[Tensor], Tensor]):
        super().__init__()
        self.register_buffer('y', y)
        self.A = A

    def err(self, x: Tensor) -> Tensor:
        return self.y - self.A(x)
    
    def set_observation(self, y: Tensor):
        self.register_buffer('y', y)

    @abstractmethod
    def log_prob(self, x: Tensor) -> Tensor:
        pass


class Gaussian(Likelihood):
    def __init__(self, y: Tensor, A: Callable[[Tensor], Tensor], std: float = 1.0):
        super().__init__(y, A)
        self.std = std

    def log_prob(self, x: Tensor, var: Tensor) -> Tensor:
        err = self.err(x)
        var_tot = self.std ** 2 + var

        return -(err ** 2 / var_tot).sum() / 2
    
    def sample(self, shape = ()) -> Tensor:
        return Normal(self.y, self.std).rsample(shape)


######################################################
################## Guidance term #####################
######################################################

class GuidedScore(ABC, nn.Module):
    def __init__(self, sde: VPSDE, likelihood: Likelihood):
        super().__init__()
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
            # err = self.likelihood.log_prob(x_, var_x0_xt)
            var_tot = self.std ** 2 + var_x0_xt
            err = -(self.err(x_) ** 2 / var_tot).sum() / 2
    
            return err, eps

        guidance, eps = grad(log_prob, has_aux=True)(x)
        return eps - sigma * guidance
    


    

if __name__ == "__main__":
    print("guidance.py")
    from experiments.kolmogorov.utils import load_data

    validset = load_data('data/kolmogorov/valid.h5', window=2)
    print(validset.shape)