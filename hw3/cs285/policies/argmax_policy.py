import torch

class ArgMaxPolicy:

    def __init__(self, critic, device):
        self.critic = critic
        self.device = device

    def get_action(self, obs):
        if len(obs.shape) > 1:
            observation = torch.tensor(obs).to(self.device)
        else:
            observation = torch.tensor(obs[None]).to(self.device)
        # TODO: pass observation to critic and use argmax of the resulting Q values as the action
        return self.critic.Q_func(observation).squeeze().argmax().item()
