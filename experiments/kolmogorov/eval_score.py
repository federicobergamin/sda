'''
File where I run the evaluation of the Kolmogorov score on a GPU cluster. I'll try to 
generate only certain figures for now.
'''

import h5py
import numpy as np
import torch

from sda.mcs import *
from sda.score import *
from sda.utils import *

from utils import *
from pathlib import Path
from tqdm import tqdm
import argparse

def main(args):
    chain = make_chain()

    # then we have to load the train score
    PATH_MODEL = Path('/scratch/fedbe') / 'sda/experiments/kolmogorov'
    score = load_score(PATH_MODEL / 'runs/earthy-sunset-3_jhd7auam/state.pth')  # k=2, width=96
    PATH_DATA = Path('/scratch/fedbe') / 'sda/kolmogorov'

    plot_folder = 'plots/'
    os.makedirs(plot_folder, exist_ok=True)

    if args.experiment == 'circle':
        def circle():
            # if you want to have an idea of how this looks like
            # you can find it here: https://colab.research.google.com/drive/1Dfli2IGpqzDDacubuZmV8llKPlVjHUg6?usp=sharing
            x = torch.linspace(-1, 1, 64)
            x = torch.cartesian_prod(x, x)

            dist = x.square().sum(dim=-1).reshape(64, 64)
            mask = torch.logical_and(0.4 < dist, dist < 0.6)

            return mask

        mask = circle().cuda()

        # now I have to define the A function
        def A(x):
            # this just compute the vorticity of x and maks show only 
            # the circle part in between. I think here we are taking the full trajectory and 
            # enforcing that the last state vorticity is a circle (correct me if I am wrong)
            # x[..., -1, :, :, :] is considering only the last state I guess
            return chain.vorticity(x[..., -1, :, :, :]) * mask
        
        # we define our VSPDE noise scheduler, where now the score
        # we are using is conditioned on the observations. In this specific example
        # we are conditioning on the observation that 
        sde = VPSDE(
            GaussianScore(
                y=0.6 * mask,
                A=A,
                std=0.2,
                sde=VPSDE(score, shape=()),
            ),
            shape=(8, 2, 64, 64),
        ).cuda()

        # here we sample from the SDE conditioned on a circle vorticity mask on the last state I guess
        x = sde.sample(steps=256, corrections=1, tau=0.5).cpu()
        w = chain.vorticity(x)

        draw(w.reshape(1, 8, 64, 64)).save('plots/x_circle.png')
        draw(w.reshape(1, 8, 64, 64), zoom=2).save('plots/x_circle_zoom_2.png')

    elif args.experiment == 'assimilation':
        # we start by considering the trajectories in the test set
        with h5py.File(PATH_DATA / 'data/test.h5') as f:
            # we pick just one trajectory
            x_star = torch.from_numpy(f['x'][1, :29])

        # we compute the vorticity of that trajectory every four step
        w = chain.vorticity(x_star[::4])

        draw(w.reshape(1, 8, 64, 64)).save('plots/x_star_assim_true_vorticity.png')
        draw(w.reshape(1, 8, 64, 64), zoom=2).save('plots/x_star_assim_true_vorticity_zoom2.png')

        # we can define our observation process
        def A(x):
            # the observation returns only a downsample representation (from 64*64 to 8*8)
            # every four steps
            return chain.coarsen(x[..., ::4, :, :, :], 8)

        # we create the observation we want to condition on.
        # it's a noisy version, but noise is pretty low
        y_star = torch.normal(A(x_star), 0.1)

        w = chain.vorticity(y_star) / 2.5
        # not sure if here we are getting back to a 64*64 version of our
        # noisy observation
        w = chain.upsample(w, 2, mode='nearest')

        # we can get the vorticity of the observed
        draw(w.reshape(1, 8, 16, 16), pad=1, zoom=4).save('plots/y_star_assim_observed_vorticity.png')
        draw(w.reshape(1, 8, 16, 16), pad=1, zoom=8).save('plots/y_star_assim_observed_vorticity_zoom_more.png')


        # now we want to compare both SDA approach to the likelihood score and the DPS approach
        sde = VPSDE(
            GaussianScore(
                y_star,
                A=A,
                std=0.1,
                sde=VPSDE(score, shape=()),
            ),
            shape=x_star.shape,
        ).cuda()

        x = sde.sample(steps=256, corrections=1, tau=0.5).cpu()
        # here we want to compute the vorticity of only the steps
        # we have observed so we can actually compute them
        w = chain.vorticity(x[::4])

        draw(w.reshape(1, 8, 64, 64)).save('plots/x_sda_assim.png')
        draw(w.reshape(1, 8, 64, 64), zoom=2).save('plots/x_sda_assim_zoom2.png')

        print('SDA performance')
        print((A(x) - y_star).std())  # should be ≈ 0.1

        sde = VPSDE(
            DPSGaussianScore(
                y_star,
                A=A,
                zeta=1.0,
                sde=VPSDE(score, shape=()),
            ),
            shape=x_star.shape,
        ).cuda()

        x = sde.sample(steps=256, corrections=1, tau=0.5).cpu()
        w = chain.vorticity(x[::4])

        draw(w.reshape(1, 8, 64, 64)).save('plots/x_dps_assim.png')
        draw(w.reshape(1, 8, 64, 64), zoom=2).save('plots/x_dps_assim_zoom2.png')

        print('DPS performance')
        print((A(x) - y_star).std())

    elif args.experiment == 'extrapolation':
        # let's start by loading the data points
        with h5py.File(PATH_DATA / 'data/test.h5') as f:
            # here we consider the 0-th trajectory and only the first 8 states
            x_star = torch.from_numpy(f['x'][0, :8])

        w = chain.vorticity(x_star)

        draw(w.reshape(1, 8, 64, 64)).save('plots/x_star_extrapolation_true_states.png')
        draw(w.reshape(1, 8, 64, 64), zoom=2).save('plots/x_star_extrapolation_true_states_zoom2.png')

        # let's define the observation process
        def A(x):
            # in this case we observe only a small square in the center
            # of the states and also every 3 steps
            return chain.coarsen(x, 4)[..., ::3, :, 4:12, 4:12]
        
        # I can compute the vorticity of the observation we are conditioning on
        y_star = torch.normal(A(x_star), 0.01)
        w = chain.vorticity(chain.coarsen(x_star, 4)) / 2

        mask = np.zeros((1, 8, 16, 16), dtype=bool)
        mask[:, ::3, 4:12, 4:12] = True

        draw(w.reshape(1, 8, 16, 16), mask, pad=1, zoom=4).save('plots/y_star_extrapolation_true_vort.png')
        draw(w.reshape(1, 8, 16, 16), mask, pad=1, zoom=8).save('plots/y_star_extrapolation_true_vort_zoom_more.png')

        # and as before we are going to compare both SDA and DPS
        sde = VPSDE(
            GaussianScore(
                y_star,
                A=A,
                std=0.01,
                sde=VPSDE(score, shape=()),
            ),
            shape=x_star.shape,
        ).cuda()

        x = sde.sample(steps=256, corrections=1, tau=0.5).cpu()
        w = chain.vorticity(x)

        draw(w.reshape(1, 8, 64, 64)).save('plots/x_sda_extra.png')
        draw(w.reshape(1, 8, 64, 64), zoom=2).save('plots/x_sda_extra_zoom2.png')

        print('SDA performance')
        print((A(x) - y_star).std())  # should be ≈ 0.01

        sde = VPSDE(
            DPSGaussianScore(
                y_star,
                A=A,
                zeta=1.0,
                sde=VPSDE(score, shape=()),
            ),
            shape=x_star.shape,
        ).cuda()

        x = sde.sample(steps=256, corrections=1, tau=0.5).cpu()
        w = chain.vorticity(x)

        draw(w.reshape(1, 8, 64, 64)).save('plots/x_dps_extra.png')
        draw(w.reshape(1, 8, 64, 64), zoom=2).save('plots/x_dps_extra_zoom2.png')
        
        print('DPS performance')
        print((A(x) - y_star).std())

    elif args.experiment == 'non-lin':
        # let's start by getting the test trajectories
        with h5py.File(PATH_DATA / 'data/test.h5') as f:
            # here we are taking the first eight states of the second trajectory
            x_star = torch.from_numpy(f['x'][2, :8])

        print('x_star shape')
        print(x_star.shape)
        # compute the vorticity
        w = chain.vorticity(x_star)
        print('w shape')
        print(w.shape)

        draw(w.reshape(1, 8, 64, 64)).save('plots/x_star_non_lin_saturation.png')
        draw(w.reshape(1, 8, 64, 64), zoom=2).save('plots/x_star_non_lin_saturation_zoom2.png')

        # define the observation process
        def A(x):
            # we get a coarsen version of the state every three steps
            x = chain.coarsen(x[..., ::3, :, :, :], 4)
            w = chain.vorticity(x)
            w = w / (1 + abs(w))

            # here we consider only part of the state observation
            return w[..., 2:14, 2:14]
        
        y_star = torch.normal(A(x_star), 0.05)
        print('y_star shape')
        print(y_star.shape)

        w = chain.vorticity(chain.coarsen(x_star, 4))
        w = w / (1 + abs(w))

        mask = np.zeros((1, 8, 16, 16), dtype=bool)
        mask[..., ::3, 2:14, 2:14] = True

        draw(w.reshape(1, 8, 16, 16), mask, pad=1, zoom=4).save('plots/y_star_non_lin_saturation.png')
        draw(w.reshape(1, 8, 16, 16), mask, pad=1, zoom=8).save('plots/y_star_non_lin_saturation_zoom_more.png')

        # as before we compare SDA and DPS
        sde = VPSDE(
            GaussianScore(
                y_star,
                A=A,
                std=0.05,
                sde=VPSDE(score, shape=()),
            ),
            shape=x_star.shape,
        ).cuda()

        x = sde.sample(steps=512, corrections=1, tau=0.5).cpu()
        w = chain.vorticity(x)

        draw(w.reshape(1, 8, 64, 64)).save('plots/x_sda_non_lin_saturation.png')
        draw(w.reshape(1, 8, 64, 64), zoom=2).save('plots/x_sda_non_lin_saturation_zoom2.png')

        print('SDA performance')
        print((A(x) - y_star).std())

        sde = VPSDE(
            DPSGaussianScore(
                y_star,
                A=A,
                zeta=1.0,
                sde=VPSDE(score, shape=()),
            ),
            shape=x_star.shape,
        ).cuda()

        x = sde.sample(steps=512, corrections=1, tau=0.5).cpu()
        w = chain.vorticity(x)

        draw(w.reshape(1, 8, 64, 64)).save('plots/x_dps_non_lin_saturation.png')
        draw(w.reshape(1, 8, 64, 64), zoom=2).save('plots/x_dps_non_lin_saturation_zoom2.png')

        print('DPS performance')
        print((A(x) - y_star).std())

    elif args.experiment == 'subsampling':
        subsampling_value = args.subsampling_val

        with h5py.File(PATH_DATA / 'data/test.h5') as f:
            # first eight states of the third trajectory
            x_star = torch.from_numpy(f['x'][3, :8])

        w = chain.vorticity(x_star)

        draw(w.reshape(1, 8, 64, 64)).save('plots/x_star_subsampling_exp.png')
        draw(w.reshape(1, 8, 64, 64), zoom=2).save('plots/x_star_subsampling_exp_zoom2.png')

        # we can define the observation model
        def A(x):
            # just observing for all the states every two pixels
            return x[..., ::subsampling_value, ::subsampling_value]
        
        y_star = torch.normal(A(x_star), 0.1)

        w = chain.vorticity(x_star) / 2

        mask = np.zeros((1, 8, 64, 64), dtype=bool)
        mask[..., ::subsampling_value, ::subsampling_value] = True

        draw(w.reshape(1, 8, 64, 64), mask).save(f'plots/y_star_subsampling_{subsampling_value}.png')
        draw(w.reshape(1, 8, 64, 64), mask, zoom=2).save(f'plots/y_star_subsampling_{subsampling_value}_zoom2.png')

        # now we have to compare SDA and DSP
        sde = VPSDE(
            GaussianScore(
                y_star,
                A=A,
                std=0.1,
                sde=VPSDE(score, shape=()),
            ),
            shape=x_star.shape,
        ).cuda()

        x = sde.sample(steps=256, corrections=1, tau=0.5).cpu()
        w = chain.vorticity(x)

        draw(w.reshape(1, 8, 64, 64)).save(f'plots/x_sda_subsampling_{subsampling_value}.png')
        draw(w.reshape(1, 8, 64, 64), zoom=2).save(f'plots/x_sda_subsampling_{subsampling_value}_zoom2.png')

        sde = VPSDE(
            DPSGaussianScore(
                y_star,
                A=A,
                zeta=1.0,
                sde=VPSDE(score, shape=()),
            ),
            shape=x_star.shape,
        ).cuda()

        x = sde.sample(steps=256, corrections=1, tau=0.5).cpu()
        w = chain.vorticity(x)

        draw(w.reshape(1, 8, 64, 64)).save(f'plots/x_dps_subsampling_{subsampling_value}.png')
        draw(w.reshape(1, 8, 64, 64), zoom=2).save(f'plots/x_dps_subsampling_{subsampling_value}_zoom2.png')

    elif args.experiment == 'loop':
        # I have to understand what is going on here
        # NOTE: I an not able to run this one
        sde = VPSDE(
            GaussianScore(
                torch.zeros(2, 64, 64),
                A=lambda x: x[:, 0] - x[:, -1],
                std=0.01,
                sde=VPSDE(score, shape=()),
                gamma=0.1,
            ),
            shape=(127, 2, 64, 64),
        ).cuda()

        x = sde.sample(steps=256, corrections=1, tau=0.5).cpu()
        w = chain.vorticity(x[::2])

        draw(w.reshape(8, 8, 64, 64)).save('plots/x_loop.png')
        draw(w.reshape(8, 8, 64, 64), zoom=2).save('plots/x_loop_zoom2.png')

    elif args.experiment == 'forecast':
        '''
        Version 0.0.1 of forecasting. 
        TODO: [] have a forecast windows --> decide if all at once or forecasting in an autoregressive way
        '''
        moving_forecast_window = args.moving_forecast_window
        trajectory_length = args.trajectory_length
        conditioned_frame = args.conditioned_frame

        if moving_forecast_window:
            # here I have to implement forecasting in a autoregressive way.
            # Therefore we sample always one frame conditioned on the last 
            # [conditioned_frame]. Therefore we condition on also the last sampled
            # frames from our model.

            # let's start by loading the data
            with h5py.File(PATH_DATA / 'data/test.h5') as f:
                # consider the first 15 states of the tenth trajectory
                x_star = torch.from_numpy(f['x'][10, :trajectory_length])

            # for i in range(frame_to_be_generated):

            raise NotImplementedError(f'Still under construction for moving_forecast_window')

        else:
            # trying to implement a naive way of forecasting, can be completely wrong to be honest, but let'see
            # also at the moment it's not autoregressive, so we are not using the space in a very smart way.
            # See this as version 0.0.1
            with h5py.File(PATH_DATA / 'data/test.h5') as f:
                # consider the first 15 states of the tenth trajectory
                x_star = torch.from_numpy(f['x'][10, :trajectory_length]) # this should be something like  (15, 2, 64,64)
            
            print(x_star.shape) #(trajectory_length, 2, 64, 64)

            if trajectory_length > 10:
                vorticity_true_trajectory = chain.vorticity(x_star[::(trajectory_length//10)])
                draw(vorticity_true_trajectory.reshape(1, 10, 64, 64)).save(f'plot_trying_stuff/forecast_true_vorticity_traj_len_{trajectory_length}_cond_frame_{conditioned_frame}.png')
                draw(vorticity_true_trajectory.reshape(1, 10, 64, 64), zoom=2).save(f'plot_trying_stuff/forecast_true_vorticity_traj_len_{trajectory_length}_cond_frame_{conditioned_frame}_zoom2.png')
            else:
                vorticity_true_trajectory = chain.vorticity(x_star)
                draw(vorticity_true_trajectory.reshape(1, trajectory_length, 64, 64)).save(f'plot_trying_stuff/forecast_true_vorticity_traj_len_{trajectory_length}_cond_frame_{conditioned_frame}.png')
                draw(vorticity_true_trajectory.reshape(1, trajectory_length, 64, 64), zoom=2).save(f'plot_trying_stuff/forecast_true_vorticity_traj_len_{trajectory_length}_cond_frame_{conditioned_frame}_zoom2.png')

            if args.save_gif:
                vorticity_true_trajectory = chain.vorticity(x_star)
                save_gif(vorticity_true_trajectory.reshape(trajectory_length, 64, 64), f'plot_trying_stuff/forecast_true_vorticity_traj_len_{trajectory_length}_cond_frame_{conditioned_frame}.gif')
            # now I have to define my A(x) and mask I guess
            # mask = np.ones((conditioned_frame, 64, 64), dtype=bool)
            # mask[:, :conditioned_frame, :, :] = True
            # print(mask[1,0:5,0:5])

            def A(x):
                # let's say that as initial example I am conditioning on 
                # the vorticity of the first two states
                # TODO: Conditioning on different stuff
                return chain.vorticity(x[..., :conditioned_frame, :, :, :]) #* mask
            
            print('Conditioned frames shapes')
            print(A(x_star).shape)
            y_star = torch.normal(A(x_star), 0.05)

            
            print('y_star shape')
            print(y_star.shape)
            draw(y_star.reshape(1,  conditioned_frame, 64, 64)).save(f'plot_trying_stuff/forecast_conditioned_true_vorticity_traj_len_{trajectory_length}_cond_frame_{conditioned_frame}.png')
            draw(y_star.reshape(1, conditioned_frame, 64, 64), zoom=2).save(f'plot_trying_stuff/forecast_conditioned_true_vorticity_traj_len_{trajectory_length}_cond_frame_{conditioned_frame}_zoom2.png')


            # now I can try both SDA and DSP and check if I can get something interesting out
            sde = VPSDE(
                GaussianScore(
                    y_star,
                    A=A,
                    std=0.1,
                    sde=VPSDE(score, shape=()),
                ),
                shape=(trajectory_length, 2, 64, 64),
            ).cuda()

            x = sde.sample(steps=512, corrections=1, tau=0.5).cpu()

            if trajectory_length > 10:
                vorticity_sda = chain.vorticity(x[::(trajectory_length//10)])
                draw(vorticity_sda.reshape(1, 10, 64, 64)).save(f'plot_trying_stuff/forecast_sda_vorticity_traj_len_{trajectory_length}_cond_frame_{conditioned_frame}.png')
                draw(vorticity_sda.reshape(1, 10, 64, 64), zoom=2).save(f'plot_trying_stuff/forecast_sda_vorticity_traj_len_{trajectory_length}_cond_frame_{conditioned_frame}_zoom2.png')
            else:
                vorticity_sda = chain.vorticity(x)
                draw(vorticity_sda.reshape(1, trajectory_length, 64, 64)).save(f'plot_trying_stuff/forecast_sda_vorticity_traj_len_{trajectory_length}_cond_frame_{conditioned_frame}.png')
                draw(vorticity_sda.reshape(1, trajectory_length, 64, 64), zoom=2).save(f'plot_trying_stuff/forecast_sda_vorticity_traj_len_{trajectory_length}_cond_frame_{conditioned_frame}_zoom2.png')
        
            if args.save_gif:
                vorticity_sda = chain.vorticity(x)
                save_gif(vorticity_sda.reshape(trajectory_length, 64, 64), f'plot_trying_stuff/forecast_sda_vorticity_traj_len_{trajectory_length}_cond_frame_{conditioned_frame}.gif')

            # now I can train DSP
            sde = VPSDE(
                DPSGaussianScore(
                    y_star,
                    A=A,
                    zeta=1.0,
                    sde=VPSDE(score, shape=()),
                ),
                shape=(trajectory_length, 2, 64, 64),
            ).cuda()

            x = sde.sample(steps=512, corrections=1, tau=0.5).cpu()

            if trajectory_length > 10:
                vorticity_dsp = chain.vorticity(x[::(trajectory_length//10)])
            else:
                vorticity_dsp = chain.vorticity(x)

            if trajectory_length > 10:
                vorticity_dsp = chain.vorticity(x[::(trajectory_length//10)])
                draw(vorticity_dsp.reshape(1, 10, 64, 64)).save(f'plot_trying_stuff/forecast_dsp_vorticity_traj_len_{trajectory_length}_cond_frame_{conditioned_frame}.png')
                draw(vorticity_dsp.reshape(1, 10, 64, 64), zoom=2).save(f'plot_trying_stuff/forecast_dsp_vorticity_traj_len_{trajectory_length}_cond_frame_{conditioned_frame}_zoom2.png')
            else:
                vorticity_dsp = chain.vorticity(x)
                draw(vorticity_dsp.reshape(1, trajectory_length, 64, 64)).save(f'plot_trying_stuff/forecast_dsp_vorticity_traj_len_{trajectory_length}_cond_frame_{conditioned_frame}.png')
                draw(vorticity_dsp.reshape(1, trajectory_length, 64, 64), zoom=2).save(f'plot_trying_stuff/forecast_dsp_vorticity_traj_len_{trajectory_length}_cond_frame_{conditioned_frame}_zoom2.png')

            if args.save_gif:
                vorticity_dsp = chain.vorticity(x)
                save_gif(vorticity_sda.reshape(trajectory_length, 64, 64), f'plot_trying_stuff/forecast_dsp_vorticity_traj_len_{trajectory_length}_cond_frame_{conditioned_frame}.gif')

    else:
        raise NotImplementedError(f'We do not have an implementation for {args.experiment} experiment')
    

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description='Evaluate Kolmogorov Experiments')
    # parser.add_argument('--seed', '-seed', type=int, default=0, help='seed for randomness generator')
    parser.add_argument('--seed', '-s', type=int, default=77, help='seed')
    parser.add_argument('--experiment', '-exp', type=str, default='circle', help='Possibilities: circle, assimilation, extrapolation, non-lin, subsampling, loop')
    parser.add_argument('--subsampling_val', '-sub_val', type=int, default=2, help='Value for subsampling in the subsampling exp')
    
    # arguments for forecasting
    parser.add_argument('--conditioned_frame', '-cond_frame', type=int, default=2, help='Number of frames we conditioned on for forecasting')
    parser.add_argument('--trajectory_length', '-traj_len', type=int, default=10, help='Trajectory length')
    parser.add_argument('--moving_forecast_window', '-moving_wind', default=False, type=bool, help='Forecasting with a moving window of size [conditioned_frame]. Corresponds to forecast in an autoregressive way.')
    parser.add_argument('--save_gif', '-gif', default=False, type=bool, help='Create a gif for the Kolmogorov systems')
    
    args = parser.parse_args()
    main(args)

