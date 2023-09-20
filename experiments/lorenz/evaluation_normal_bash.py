'''
File where I try to rewrite the code that is using the dawgz
library in the usual way that support bash scripts.
'''

from h5py import *
from typing import *

from sda.mcs import *
from sda.score import *
from sda.utils import *

from utils import *
import argparse
from pathlib import Path

def main(args):
    # DATA_PATH = Path('../experiments/lorenz/data')
    TRAINED_MODEL_PATH = Path('/scratch/fedbe/sda/experiments/lorenz')
    RESULTS_PATH = Path('../experiments/lorenz/results')
    # RESULTS_PATH = Path('results')

    # here depending on the window size I have to select the correct
    # run. Not sure I really enjoyed this, but for now I can live with
    if args.experiment == 'local':
        if args.window // 2 == 1:
            run_name = 'fanciful-haze-28_xzgz3m5k'
        elif args.window // 2 == 2:
            run_name = 'fast-universe-25_gohvbo6v'
        elif args.window // 2 == 3:
            run_name = 'effortless-serenity-27_dbfmmhb6'
        elif args.window // 2 == 4:
            run_name = 'efficient-plasma-29_mdemiyni'
        else:
            raise NotImplementedError('We did not consider k > 4 in this experiment')
    else:
        run_name = 'generous-lake-11_g7bebatj'

    
    for freq in ['lo', 'hi']:

        for i in range(64):
            # I have to make a chain
            chain = make_chain() 

            # load the observation 
            with h5py.File(RESULTS_PATH / 'obs.h5', mode='r') as f:
                y = torch.from_numpy(f[freq][i])

            A = lambda x: chain.preprocess(x)[..., :1]

            if freq == 'lo':  # low frequency & low noise
                sigma, step = 0.05, 8
            else:             # high frequency & high noise
                sigma, step = 0.25, 1

            x = posterior(y, A=A, sigma=sigma, step=step)[:1024]
            x_ = posterior(y, A=A, sigma=sigma, step=step)[:1024]

            log_px = log_prior(x).mean().item()
            log_py = log_likelihood(y, x, A=A, sigma=sigma, step=step).mean().item()
            w1 = emd(x, x_).item()

            with open(RESULTS_PATH / f'stats_{freq}.csv', mode='a') as f:
                f.write(f'{i},ground-truth,,{log_px},{log_py},{w1}\n')

            print('GT:', log_px, log_py, w1, flush=True)

            # Score
            if args.experiment == 'local':
                score = load_score(TRAINED_MODEL_PATH / f'runs/{run_name}/state.pth', local=True)
            else:
                score = load_score(TRAINED_MODEL_PATH / f'runs/{run_name}/state.pth', local=False)

            if not args.experiment == 'local':
                score = MCScoreWrapper(score)

            sde = VPSDE(
                GaussianScore(
                    y=y,
                    A=lambda x: x[..., ::step, :1],
                    std=sigma,
                    sde=VPSDE(score, shape=()),
                ),
                shape=(65, 3),
            ).cuda()

            for C in (0, 1, 2, 4, 8, 16):
                x = sde.sample((1024,), steps=256, corrections=C, tau=0.25).cpu()
                x = chain.postprocess(x)

                log_px = log_prior(x).mean().item()
                log_py = log_likelihood(y, x, A=A, sigma=sigma, step=step).mean().item()
                w1 = emd(x, x_).item()

                with open(RESULTS_PATH / f'stats_{freq}.csv', mode='a') as f:
                    f.write(f'{i},{run_name},{C},{log_px},{log_py},{w1}\n')

                print(f'{C:02d}:', log_px, log_py, w1, flush=True)


if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description='Train Lorenz Experiments')
    # parser.add_argument('--seed', '-seed', type=int, default=0, help='seed for randomness generator')
    parser.add_argument('--seed', '-s', type=int, default=77, help='seed')
    parser.add_argument('--experiment', '-exp', type=str, default='local', help='local: k <= 4, global: k>4')
    parser.add_argument('--window', '-w', type=int, default=5, help='seed')
    args = parser.parse_args()
    main(args)
