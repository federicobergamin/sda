from abc import ABC, abstractmethod

from torch import Tensor, nn


# class ForwardOperator(ABC):
#     def __init__(self):
#         pass

#     @abstractmethod
#     def __call__(self, x: Tensor) -> Tensor:
#         pass


class Likelihood(ABC, nn.Module):
    def __init__(self, y: Tensor, A: callable[Tensor, Tensor]):
        self.register_buffer('y', y)
        self.A = A

    def err(self, x: Tensor) -> Tensor:
        return self.y - self.A(x)

    @abstractmethod
    def log_prob(self, x: Tensor) -> Tensor:
        pass


class Gaussian(Likelihood):
    def __init__(self, y: Tensor, A: callable[Tensor, Tensor], std: float = 1.0):
        super().__init__(y, A)
        self.std = std

    def log_prob(self, x: Tensor, var: Tensor) -> Tensor:
        err = self.err(x)
        var_tot = self.std ** 2 + var

        return -(err ** 2 / var_tot).sum() / 2