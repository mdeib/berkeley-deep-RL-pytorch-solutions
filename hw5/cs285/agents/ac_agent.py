import numpy as np
import torch
from collections import OrderedDict

from cs285.policies.MLP_policy import MLPPolicyAC
from cs285.critics.bootstrapped_continuous_critic import BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.exploration.exploration import *
from cs285.exploration.density_model import *

class ACAgent:
    def __init__(self, env, agent_params):
        super(ACAgent, self).__init__()

        self.env = env
        self.agent_params = agent_params
        self.num_critic_updates_per_agent_update = agent_params['num_critic_updates_per_agent_update']
        self.num_actor_updates_per_agent_update = agent_params['num_actor_updates_per_agent_update']
        self.device = agent_params['device']

        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']

        self.actor = MLPPolicyAC(self.agent_params['ob_dim'],
                               self.agent_params['ac_dim'],
                               self.agent_params['n_layers'],
                               self.agent_params['size'],
                               self.agent_params['device'],
                               discrete=self.agent_params['discrete'],
                               learning_rate=self.agent_params['learning_rate'],
                               )
        self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer(agent_params['replay_size'])

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        ob, next_ob, rew, done = map(lambda x: torch.from_numpy(x).to(self.device), [ob_no, next_ob_no, re_n, terminal_n])

        value = self.critic.value_func(ob).squeeze()
        next_value = self.critic.value_func(next_ob).squeeze() * (1 - done)
        adv_n = rew + (self.gamma * next_value) - value
        adv_n = adv_n.cpu().detach().numpy()

        if self.standardize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
        return adv_n

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        loss = OrderedDict()

        for critic_update in range(self.num_critic_updates_per_agent_update):
            loss['Critic_Loss'] = self.critic.update(ob_no, next_ob_no, re_n, terminal_n)

        adv_n = self.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n) # put final critic loss here

        for actor_update in range(self.num_actor_updates_per_agent_update):
            loss['Actor_Loss'] = self.actor.update(ob_no, ac_na, adv_n)  # put final actor loss here

        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)

class Exploratory_ACAgent(ACAgent):
    def __init__(self, env, agent_params):
        super().__init__(env, agent_params)
        self.dm_type = agent_params['density_model']

        ########################################################################
        # Initalize exploration density model
        if self.dm_type != 'none':
            if agent_params['env_name'] == 'PointMass-v0' and self.dm_type == 'hist':
                self.density_model = Histogram(
                    nbins = self.env.grid_size,
                    preprocessor = self.env.preprocess)
                self.exploration = DiscreteExploration(
                    density_model = self.density_model,
                    bonus_coeff = agent_params['bonus_coeff'])
            elif self.dm_type == 'rbf':
                self.density_model = RBF(sigma = agent_params['sigma'])
                self.exploration = RBFExploration(
                    density_model = self.density_model,
                    bonus_coeff = agent_params['bonus_coeff'],
                    replay_buffer = self.replay_buffer)
            elif self.dm_type == 'ex2':
                self.density_model = Exemplar(
                    ob_dim = agent_params['ob_dim'],
                    hid_dim = agent_params['density_hiddim'],
                    learning_rate = agent_params['density_lr'],
                    kl_weight = agent_params['kl_weight'],
                    device = agent_params['device'])
                self.exploration = ExemplarExploration(
                    density_model = self.density_model,
                    bonus_coeff = agent_params['bonus_coeff'],
                    train_iters = agent_params['density_train_iters'],
                    bsize = agent_params['density_batch_size'],
                    replay_buffer = self.replay_buffer)
            else:
                raise NotImplementedError

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):

        ########################################################################
        # Modify the reward to include exploration bonus
        """
            1. Fit density model
                if params["density_model"] == 'ex2':
                    the call to exploration.fit_density_model should return ll, kl, elbo
                else:
                    the call to exploration.fit_density_model should return nothing
            2. Modify the re_n with the reward bonus by calling exploration.modify_reward
        """
        ex2_vars = None
        old_re_n = re_n
        if self.dm_type == 'none':
            pass
        else:
            # 1. Fit density model
            if self.dm_type == 'ex2':
                ### PROBLEM 3
                ### YOUR CODE HERE
                ex2_vars = self.exploration.fit_density_model(ob_no)
            elif self.dm_type == 'hist' or self.dm_type == 'rbf':
                ### PROBLEM 1
                ### YOUR CODE HERE
                self.exploration.fit_density_model(ob_no)
            else:
                assert False

            # 2. Modify the reward
            ### PROBLEM 1
            ### YOUR CODE HERE
            re_n = self.exploration.modify_reward(re_n, ob_no)

            print('average state', np.mean(ob_no, axis=0))
            print('average action', np.mean(ac_na, axis=0))

        loss = OrderedDict()

        for critic_update in range(self.num_critic_updates_per_agent_update):
            loss['Critic_Loss'] = self.critic.update(ob_no, next_ob_no, re_n, terminal_n)

        adv_n = self.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n) # put final critic loss here

        for actor_update in range(self.num_actor_updates_per_agent_update):
            loss['Actor_Loss'] = self.actor.update(ob_no, ac_na, adv_n)  # put final actor loss here

        return loss, ex2_vars
