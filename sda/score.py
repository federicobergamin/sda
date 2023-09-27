r"""Score modules"""

import math
import torch
import torch.nn as nn

from torch import Size, Tensor
from tqdm import tqdm
from typing import *
from zuko.utils import broadcast

from .nn import *


class TimeEmbedding(nn.Sequential):
    r"""Creates a time embedding.

    Arguments:
        features: The number of embedding features.
    """

    def __init__(self, features: int):
        super().__init__(
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, features),
        )

        self.register_buffer('freqs', torch.pi / 2 * 1e3 ** torch.linspace(0, 1, 64))

    def forward(self, t: Tensor) -> Tensor:
        t = self.freqs * t.unsqueeze(dim=-1)
        t = torch.cat((t.cos(), t.sin()), dim=-1)

        return super().forward(t)


class ScoreNet(nn.Module):
    r"""Creates a simple score network made of residual blocks.

    Arguments:
        features: The number of features.
        embedding: The number of time embedding features.
    """

    def __init__(self, features: int, embedding: int = 16, **kwargs):
        super().__init__()

        self.embedding = TimeEmbedding(embedding)
        self.network = ResMLP(features + embedding, features, **kwargs)

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

    def __init__(self, channels: int, embedding: int = 64, **kwargs):
        super().__init__()

        self.embedding = TimeEmbedding(embedding)
        self.network = UNet(channels, channels, embedding, **kwargs)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        dims = self.network.spatial + 1

        y = x.reshape(-1, *x.shape[-dims:])
        t = t.reshape(-1)
        t = self.embedding(t)

        return self.network(y, t).reshape(x.shape)


class MCScoreWrapper(nn.Module):
    r"""Disguises a `ScoreUNet` as a score network for a Markov chain.
    
        Just a wrapper for our score network that is passed in the constructor.
        The forward just call the forward of the score network
    """

    def __init__(self, score: nn.Module):
        super().__init__()

        self.score = score

    def forward(
        self,
        x: Tensor,  # (B, L, C, H, W)
        t: Tensor,  # ()
    ) -> Tensor:
        return self.score(x.transpose(1, 2), t).transpose(1, 2)


class MCScoreNet(nn.Module):
    r"""Creates a score network for a Markov chain.

    Arguments:
        features: The number of features.
        order: The order of the Markov chain.
    """

    def __init__(self, features: int, order: int = 1, **kwargs):
        super().__init__()

        self.order = order

        if kwargs.get('spatial', 0) > 0:
            build = ScoreUNet
        else:
            build = ScoreNet

        # kernel is just the scoreNet we decide to use
        # order is just the K variable they decide to use in each experiments.
        # So if we have k=2, then for each x_i we consider x_{i-2},x_{i-1}, x_{i}, x_{i+1}, x_{i+2}
        # so we are processing 2*order + 1 frames of _features_ number of features

        self.kernel = build(features * (2 * order + 1), **kwargs)

    def forward(
        self,
        x: Tensor,  # (B, L, C, H, W) --> (batch, length_trajectory, channel, height, width) --> what is a batch in this setting???
        t: Tensor,  # ()
    ) -> Tensor:
        '''
        This forward is a smart implementation of Algorithm 2 in the paper
        '''
        # create all the pseudo markov blanket of order k, given the batch of trajectories
        x = self.unfold(x, self.order)
        # compute all the different scores. Size should be (B, number_of_blankets, number_frame_in_blankets)
        # or it should be the same as x shape, which I found it strange
        # TODO: check shape of s
        s = self.kernel(x, t)
        # compute the approximated scores
        s = self.fold(s, self.order)

        return s

    # the tag is just compiling the function when it is first called during tracing
    @staticmethod
    @torch.jit.script_if_tracing
    def unfold(x: Tensor, order: int) -> Tensor:
        '''
        This method take the batch of trajectories, and return all the psudo markov
        blanket described by Algorithm 2 in the paper.
        So it just create the following:
        - x_{1:2k+1}(t)
        - x_{i−k:i+k}(t) for i = k + 2 to L − k − 1
        - x_{L−2k:L}(t)

        These are all the input to our score network that are used to compute the approximate score.
        '''
        x = x.unfold(1, 2 * order + 1, 1)
        x = x.movedim(-1, 2)
        x = x.flatten(2, 3)

        return x

    @staticmethod
    @torch.jit.script_if_tracing
    def fold(x: Tensor, order: int) -> Tensor:
        '''
        Function that given all the scores computed in each markov blanket and
        compose the approximated score as described in Algorithm 2
        '''
        x = x.unflatten(2, (2 * order  + 1, -1))

        return torch.cat((
            x[:, 0, :order],
            x[:, :, order],
            x[:, -1, -order:],
        ), dim=1)


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


class DPSGaussianScore(nn.Module):
    r"""Creates a score module for Gaussian inverse problems.

    .. math:: p(y | x) = N(y | A(x), Σ)

    References:
        | Diffusion Posterior Sampling for General Noisy Inverse Problems (Chung et al., 2022)
        | https://arxiv.org/abs/2209.14687

    Note:
        This module returns :math:`-\sigma(t) s(x(t), t | y)`.
    """

    def __init__(
        self,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        sde: VPSDE,
        zeta: float = 1.0,
    ):
        super().__init__()

        self.register_buffer('y', y)

        self.A = A
        self.sde = sde
        self.zeta = zeta

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            eps = self.sde.eps(x, t)
            x_ = (x - sigma * eps) / mu
            err = (self.y - self.A(x_)).square().sum()

        s, = torch.autograd.grad(err, x)
        s = -s * self.zeta / err.sqrt()

        return eps - sigma * s


class GaussianScore(nn.Module):
    r"""Creates a score module for Gaussian inverse problems.

    .. math:: p(y | x) = N(y | A(x), Σ)

    Note:
        This module returns :math:`-\sigma(t) s(x(t), t | y)`.

    Comments: this is the class implementing the likelihood score 
                \nabla_{x(t)} log N (y | A(\hat{x}), \Sigma_{y} + (\sigma(t)^2 / mu(t)^2) \Gamma)
            
            where \hat{x} is computed using Tweedie's Formula given by:

            \hat{x} = \frac{x(t) + \sigma(t)^2 s_x}{\mu(t)}

            and s_x is the prior score given by the score newtork.

            Since we are using the following parametrization of the score
            function:
                    s_x = - \eps_x / \sigma(t)

            then Tweedie's formula becomes:
                 \hat{x} = \frac{x(t) - \sigma(t) \eps_x}{\mu(t)}

            which is exactly shown on line 421.
    """

    def __init__(
        self,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        std: Union[float, Tensor],
        sde: VPSDE,
        gamma: Union[float, Tensor] = 1e-2,
        detach: bool = False,
    ):
        super().__init__()

        self.register_buffer('y', y)
        self.register_buffer('std', torch.as_tensor(std))
        self.register_buffer('gamma', torch.as_tensor(gamma))

        # observation process A
        self.A = A
        # score netwerk
        self.sde = sde
        self.detach = detach

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        if self.detach:
            eps = self.sde.eps(x, t)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            if not self.detach:
                eps = self.sde.eps(x, t)
            
            # this is Tweedie's formula under the chosen parametrization
            # of epsilon
            x_ = (x - sigma * eps) / mu

            # here we apply the observation process to x_ so we can compare with the 
            # observation y we are given
            err = self.y - self.A(x_)
            var = self.std ** 2 + self.gamma * (sigma / mu) ** 2

            log_p = -(err ** 2 / var).sum() / 2

        s, = torch.autograd.grad(log_p, x)

        return eps - sigma * s
