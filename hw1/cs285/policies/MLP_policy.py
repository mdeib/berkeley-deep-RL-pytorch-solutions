import numpy as np
import torch
import torch.nn as nn

class MLPPolicy(nn.Module):

    def __init__(self,
        ac_dim,
        ob_dim,
        n_layers,
        size,
        device,
        lr = 1e-4,
        training=True,
        discrete=False, # unused for now
        nn_baseline=False, # unused for now
        **kwargs):
        super().__init__()

        # init vars
        self.training = training
        self.device = device

        # network architecture
        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(ob_dim, size))#first hidden layer
        self.mlp.append(nn.Tanh())

        for h in range(n_layers - 1): #additional hidden layers
            self.mlp.append(nn.Linear(size, size))
            self.mlp.append(nn.Tanh())

        self.mlp.append(nn.Linear(size, ac_dim)) #output layer, no activation function

        #loss and optimizer
        if self.training:
            self.loss_func = nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.parameters(), lr)

        self.to(device)

    ##################################

    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def restore(self, filepath):
        self.load_state_dict(torch.load(filepath))

    ##################################

    # query this policy with observation(s) to get selected action(s)
    def get_action(self, obs):
        if len(obs.shape)>1:
            observation = obs
        else:
            observation = obs[None]

        return self(torch.Tensor(observation).to(self.device)).cpu().detach().numpy()

    # update/train this policy
    def update(self, observations, actions):
        raise NotImplementedError

#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):

    """
        This class is a special case of MLPPolicy,
        which is trained using supervised learning.
        The relevant functions to define are included below.
    """

    def update(self, observations, actions):
        assert self.training, 'Policy must be created with training = true in order to perform training updates...'

        # TODO define network update
        self.optimizer.zero_grad()
        predicted_actions = self(torch.Tensor(observations).to(self.device))
        loss = self.loss_func(predicted_actions, torch.Tensor(actions).to(self.device))
        loss.backward()
        self.optimizer.step()

        #print("loss:", loss.item())
