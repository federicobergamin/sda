import  torch
import math
from zuko.utils import broadcast
from pathlib import Path
import h5py


def load_data(file: Path, window: int = None):
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


def main():
    window = 5
    freqs = torch.pi / 2 * 1e3 ** torch.linspace(0, 1, 64)
    #t = torch.linspace(0, 1, 2).unsqueeze(dim = 0)
    t = torch.rand(64)
    t = freqs * t.unsqueeze(dim=-1)
    t = torch.cat((t.cos(), t.sin()), dim=-1)
    print(t.shape)

    #DATA_PATH = Path('../sda_data/Kolmogorov dataset')
    #trainset = load_data(DATA_PATH / 'train.h5', window=window)
    #print(trainset.shape)

if __name__ == "__main__":
    main()