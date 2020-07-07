import torch
import numpy as np

from cs285.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer, PiecewiseSchedule
from cs285.policies.argmax_policy import ArgMaxPolicy
from cs285.critics.dqn_critic import DQNCritic


class DQNAgent(object):
    def __init__(self, env, agent_params):

        print(agent_params['optimizer_spec'])

        self.env = env
        self.agent_params = agent_params
        self.batch_size = agent_params['batch_size']
        self.device = agent_params['device']
        self.last_obs = self.env.reset()

        self.num_actions = agent_params['ac_dim']
        self.learning_starts = agent_params['learning_starts']
        self.learning_freq = agent_params['learning_freq']
        self.target_update_freq = agent_params['target_update_freq']

        self.replay_buffer_idx = None
        self.exploration = agent_params['exploration_schedule']
        self.optimizer_spec = agent_params['optimizer_spec']

        self.critic = DQNCritic(agent_params, self.optimizer_spec)
        self.actor = ArgMaxPolicy(self.critic, self.device)

        lander = agent_params['env_name'] == 'LunarLander-v2'
        self.replay_buffer = MemoryOptimizedReplayBuffer(agent_params['replay_buffer_size'], agent_params['frame_history_len'], lander=lander)
        self.t = 0
        self.num_param_updates = 0

    def add_to_replay_buffer(self, paths):
        pass

    def step_env(self):

        """
            Step the env and store the transition

            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.

            Note that self.last_obs must always point to the new latest observation.
        """

        # TODO store the latest observation into the replay buffer
        # HINT: see replay buffer's function store_frame
        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

        eps = self.exploration.value(self.t)
        # TODO use epsilon greedy exploration when selecting action
        # HINT: take random action
            # with probability eps (see np.random.random())
            # OR if your current step number (see self.t) is less that self.learning_starts
        perform_random_action = (self.t < self.learning_starts) or (np.random.random() < eps)

        if perform_random_action:
            action = np.random.randint(self.num_actions)
        else:
            # TODO query the policy to select action
            # HINT: you cannot use "self.last_obs" directly as input
            # into your network, since it needs to be processed to include context
            # from previous frames.
            # Check out the replay buffer, which has a function called
            # encode_recent_observation that will take the latest observation
            # that you pushed into the buffer and compute the corresponding
            # input that should be given to a Q network by appending some
            # previous frames.
            enc_last_obs = self.replay_buffer.encode_recent_observation()
            enc_last_obs = torch.tensor(enc_last_obs[None, :]).to(self.device)

            # TODO query the policy with enc_last_obs to select action
            action = self.actor.get_action(enc_last_obs)

        # TODO take a step in the environment using the action from the policy
        # HINT1: remember that self.last_obs must always point to the newest/latest observation
        # HINT2: remember the following useful function that you've seen before:
            #obs, reward, done, info = env.step(action)
        obs, reward, done, info = self.env.step(action)

        # TODO store the result of taking this action into the replay buffer
        # HINT1: see replay buffer's store_effect function
        # HINT2: one of the arguments you'll need to pass in is self.replay_buffer_idx from above
        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        # TODO if taking this step resulted in done, reset the env (and the latest observation)
        if done:
            self.last_obs = self.env.reset()
        else:
            self.last_obs = obs

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [],[],[],[],[]

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):

        """
            Here, you should train the DQN agent.
            This consists of training the critic, as well as periodically updating the target network.
        """
        loss = 0
        if (self.t > self.learning_starts and \
                self.t % self.learning_freq == 0 and \
                self.replay_buffer.can_sample(self.batch_size)):

            # TODO populate the parameters and implement actor.update()
            loss = self.critic.update(ob_no, ac_na, re_n, next_ob_no, terminal_n)

            # TODO: load newest parameters into the target network
            if self.num_param_updates % self.target_update_freq == 0:
                self.critic.target_Q_func.load_state_dict(self.critic.Q_func.state_dict())

            self.num_param_updates += 1

        self.t += 1

        return loss
