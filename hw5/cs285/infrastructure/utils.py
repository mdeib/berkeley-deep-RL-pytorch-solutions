import numpy as np
import time
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self,
        input_dim,
        output_dim,
        n_layers,
        size,
        device,
        discrete,
        activation = nn.Tanh()):
        super().__init__()

        self.discrete = discrete

        # network architecture
        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(input_dim, size)) #first hidden layer
        self.mlp.append(activation)

        for h in range(n_layers - 1): #additional hidden layers
            self.mlp.append(nn.Linear(size, size))
            self.mlp.append(activation)

        self.mlp.append(nn.Linear(size, output_dim)) #output layer, no activation function

        #if continuous define logstd variable
        if not self.discrete:
            self.logstd = nn.Parameter(torch.zeros(output_dim))

        self.to(device)

    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        if self.discrete:
            return x
        else:
            return (x, self.logstd.exp())

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def restore(self, filepath):
        self.load_state_dict(torch.load(filepath))

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, animate, itr):
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            #animate_this_episode = (len(paths) == 0 and (itr % 10 == 0) and animate)
            animate_this_episode = (len(paths) == 0 and animate)
            path = sample_trajectory(env, policy, max_path_length, animate_this_episode)
            paths.append(path)
            timesteps_this_batch += get_pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        return paths, timesteps_this_batch

def sample_trajectory(env, policy, max_path_length, animate_this_episode):
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals = [], [], [], [], []
    steps = 0
    while True:
        if animate_this_episode:
            env.render()
            time.sleep(0.1)

        obs.append(ob)
        ac = policy.get_action(ob)
        acs.append(ac)

        ob, rew, done, _ = env.step(ac)

        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        if done or steps > max_path_length:
            terminals.append(1)
            break
        else:
            terminals.append(0)

    path = {"observation" : np.array(obs, dtype=np.float32),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}

    return path

def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards

def get_pathlength(path):
    return len(path["reward"])
