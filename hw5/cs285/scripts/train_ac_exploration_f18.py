#If not using anaconda use next two lines:
import sys
sys.path.append(r'C:\Users\Matt\OneDrive\RL\UCBerkeley-deep-RL\hw5')

import numpy as np
import gym
import os
import torch
import time
from multiprocessing import Process
from cs285.infrastructure.rl_trainer import RL_Trainer
from cs285.agents.ac_agent import Exploratory_ACAgent

def train_AC(params):

    computation_graph_args = {
        'n_layers': params['n_layers'],
        'size': params['size'],
        'device': params['device'],
        'learning_rate': params['learning_rate'],
        'num_target_updates': params['num_target_updates'],
        'num_grad_steps_per_target_update': params['num_grad_steps_per_target_update'],
        }

    train_args = {
        'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
        'num_critic_updates_per_agent_update': params['num_critic_updates_per_agent_update'],
        'num_actor_updates_per_agent_update': params['num_actor_updates_per_agent_update'],
        'gamma': params['discount'],
        'standardize_advantages': not(params['dont_standardize_advantages']),
    }

    exploration_args = {
        'density_model': params['density_model'],
        'bonus_coeff': params['bonus_coeff'],
        'kl_weight': params['kl_weight'],
        'density_lr': params['density_lr'],
        'density_train_iters': params['density_train_iters'],
        'density_batch_size': params['density_batch_size'],
        'density_hiddim': params['density_hiddim'],
        'replay_size': params['replay_size'],
        'sigma': params['sigma'],
    }

    params['agent_params'] = {**computation_graph_args, **train_args, **exploration_args}
    params['agent_class'] = Exploratory_ACAgent

    rl_trainer = RL_Trainer(params)
    rl_trainer.run_training_loop(params['n_iter'], policy = rl_trainer.agent.actor)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vac')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--dont_standardize_advantages', '-dsa', action='store_true')
    parser.add_argument('--num_target_updates', '-ntu', type=int, default=10)
    parser.add_argument('--num_grad_steps_per_target_update', '-ngsptu', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=32)
    parser.add_argument('--scalar_log_freq', type=int, default=1) #-1 to disable
    parser.add_argument('--use_gpu', '-gpu', default = True) #-1 to disable
    parser.add_argument('--which_gpu', default = 0) #-1 to disable
    ########################################################################
    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--num_critic_updates_per_agent_update', type=int, default=1)
    parser.add_argument('--num_actor_updates_per_agent_update', type=int, default=1)
    ########################################################################
    parser.add_argument('--bonus_coeff', '-bc', type=float, default=1e-3)
    parser.add_argument('--density_model', type=str, default='hist | rbf | ex2 | none')
    parser.add_argument('--kl_weight', '-kl', type=float, default=1e-2)
    parser.add_argument('--density_lr', '-dlr', type=float, default=5e-3)
    parser.add_argument('--density_train_iters', '-dti', type=int, default=1000)
    parser.add_argument('--density_batch_size', '-db', type=int, default=64)
    parser.add_argument('--density_hiddim', '-dh', type=int, default=32)
    parser.add_argument('--replay_size', '-rs', type=int, default=int(1e6))
    parser.add_argument('--sigma', '-sig', type=float, default=0.2)
    ########################################################################

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    if torch.cuda.is_available() and params["use_gpu"]:
        which_gpu = "cuda:" + str(params["which_gpu"])
        params["device"] = torch.device(which_gpu)
        print("Pytorch is running on GPU", params["which_gpu"])
    else:
        params["device"] = torch.device("cpu")
        print("Pytorch is running on the CPU")

    logdir_prefix = 'ac_'

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    params['max_path_length'] = params['ep_len'] if params['ep_len'] > 0 else None

    processes = []
    master_seed = params["seed"]

    for e in range(args.n_experiments):
        params['seed'] = master_seed + 10 * e

        logdir = 'seed' + str(params['seed']) + '_' + logdir_prefix + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
        logdir = os.path.join(data_path, logdir)
        if not(os.path.exists(logdir)):
            os.makedirs(logdir)
        params['logdir'] = logdir
        print('Running experiment with seed %d' %params['seed'])

        p = Process(target = train_AC(params), args = tuple())
        p.start()
        processes.append(p)
        # if you comment in the line below, then the loop will block
        # until this process finishes
        # p.join()

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
