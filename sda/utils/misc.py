r"""Helpers"""

import h5py
import json
import math
import numpy as np
import ot
import random
import torch

from pathlib import Path
from torch import Tensor
from tqdm import trange
from typing import *

from ..score import *


def random_config(configs: Dict[str, Sequence[Any]]) -> Dict[str, Any]:
    return {
        key: random.choice(values)
        for key, values in configs.items()
    }


def save_config(config: Dict[str, Any], path: Path) -> None:
    with open(path / 'config.json', mode='x') as f:
        json.dump(config, f)


def load_config(path: Path) -> Dict[str, Any]:
    with open(path / 'config.json', mode='r') as f:
        return json.load(f)


def save_data(x: Tensor, file: Path) -> None:
    with h5py.File(file, mode='w') as f:
        f.create_dataset('x', data=x, dtype=np.float32)


def load_data(file: Path, window: int = None) -> Tensor:
    '''
    The window argument prepared the pseudo markov blanket
    used for approxiamting the score. However I think there is something
    strange going on here. Or I am just confused.

    NOTE: If we condier a specific window size, shouldn't we be 
    symmetric. So I have the feeling that data = data.unfold(1, window, 1)
    should be data = data.unfold(1, 2*window+1, 1). But in this case it's not the case.

    NOTE (2): windows should be an odd number only, otherwise it is not running. Or at least 
                this is what is happening in the lorenz experiment. So k = window // 2
                
    '''
    with h5py.File(file, mode='r') as f:
        data = f['x'][:]

    data = torch.from_numpy(data)

    if window is None:
        pass
    elif window == 1:
        data = data.flatten(0, 1)
    else:
        data = data.unfold(1, window, 1)
        data = data.movedim(-1, 2)
        data = data.flatten(2, 3)
        data = data.flatten(0, 1)

    return data


def bpf(
    x: Tensor,  # (M, *)
    y: Tensor,  # (N, *)
    transition: Callable[[Tensor], Tensor],
    likelihood: Callable[[Tensor, Tensor], Tensor],
    step: int = 1,
) -> Tensor:  # (M, N + 1, *)
    r"""Performs bootstrap particle filter (BPF) sampling

    .. math:: p(x_0, x_1, ..., x_n | y_1, ..., y_n)
        = p(x_0) \prod_i p(x_i | x_{i-1}) p(y_i | x_i)

    Wikipedia:
        https://wikipedia.org/wiki/Particle_filter

    Arguments:
        x: A set of initial states :math:`x_0`.
        y: The vector of observations :math:`(y_1, ..., y_n)`.
        transition: The transition function :math:`p(x_i | x_{i-1})`.
        likelihood: The likelihood function :math:`p(y_i | x_i)`.
        step: The number of transitions per observation.
    """

    x = x[:, None]

    for yi in y:
        for _ in range(step):
            xi = transition(x[:, -1])
            x = torch.cat((x, xi[:, None]), dim=1)

        w = likelihood(yi, xi)
        j = torch.multinomial(w, len(w), replacement=True)
        x = x[j]

    return x

