import time

from collections import OrderedDict
import pickle
import numpy as np
import torch
import gym
import os
import sys
from gym import wrappers

import cs285.envs #register all of our envs
from cs285.infrastructure.utils import *
from cs285.infrastructure.logger import Logger

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40 # we overwrite this in the code below

class RL_Trainer(object):

    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params, create logger, create TF session
        self.params = params
        self.logger = Logger(self.params['logdir'])

        # Set random seeds
        seed = self.params['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)

        #############
        ## ENV
        #############

        # Make the gym environment
        if self.params['env_name'] == 'PointMass-v0':
            from cs285.envs.pointmass import PointMass
            self.env = PointMass()
        else:
            self.env = gym.make(self.params['env_name'])
        self.env.seed(seed)
        self.params['agent_params']['env_name'] = self.params['env_name']

        self.max_path_length = self.params['max_path_length'] or self.env.spec.max_episode_steps

        # Maximum length for episodes
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps

        # Is this env continuous, or self.discrete?
        self.params['agent_params']['discrete'] = isinstance(self.env.action_space, gym.spaces.Discrete)

        # Observation and action sizes
        self.params['agent_params']['ob_dim'] = self.env.observation_space.shape[0]
        self.params['agent_params']['ac_dim'] = self.env.action_space.n if self.params['agent_params']['discrete'] else self.env.action_space.shape[0]

        #############
        ## AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])


    def run_training_loop(self, n_iter, policy):

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)

            # decide if metrics should be logged
            if self.params['scalar_log_freq'] == -1:
                self.logmetrics = False
            elif itr % self.params['scalar_log_freq'] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False

            # collect trajectories, to be used for training
            paths, envsteps_this_batch = self.collect_training_trajectories(itr, policy, self.params['batch_size'])

            self.total_envsteps += envsteps_this_batch

            # add collected data to replay buffer
            self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            loss, ex2_vars = self.train_agent()

            # log/save
            if self.logmetrics:
                # perform logging
                print('\nBeginning logging procedure...')
                self.perform_logging(itr, paths, policy, loss, ex2_vars)

    ####################################
    ####################################

    def collect_training_trajectories(self, itr, policy, batch_size):
        print("\nCollecting data to be used for training...")
        paths, envsteps_this_batch = sample_trajectories(self.env, policy, batch_size, self.max_path_length, self.params['render'], itr)

        return paths, envsteps_this_batch

    def train_agent(self):
        #print('\nTraining agent using sampled data from replay buffer...')
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['batch_size'])

            loss, ex2_vars = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
        return loss, ex2_vars

    ####################################

    def perform_logging(self, itr, paths, eval_policy, loss, ex2_vars):

        if self.logmetrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]

            # decide what to log
            logs = OrderedDict()

            if ex2_vars != None:
                logs["Log_Likelihood_Average"] = np.mean(ex2_vars[0])
                logs["KL_Divergence_Average"] = np.mean(ex2_vars[1])
                logs["ELBO_Average"] = np.mean(ex2_vars[2])

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            if isinstance(loss, dict):
                logs.update(loss)
            else:
                logs["Training loss"] = loss

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()
