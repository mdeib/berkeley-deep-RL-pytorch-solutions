from cs285.infrastructure.utils import normalize, unnormalize, MLP
import numpy as np
import torch
from torch import nn

class FFModel:
    def __init__(self, ac_dim, ob_dim, n_layers, size, device, learning_rate = 0.001):
        # init vars
        self.device = device

        #TODO - specify ouput dim and input dim of delta func MLP
        self.delta_func = MLP(input_dim = ob_dim + ac_dim,
                              output_dim = ob_dim,
                              n_layers = n_layers,
                              size = size,
                              device = self.device,
                              discrete = True)

        #TODO - define the delta func optimizer. Adam optimizer will work well.
        self.optimizer = torch.optim.Adam(self.delta_func.parameters(), lr = learning_rate)

    #############################

    def get_prediction(self, obs, acs, data_statistics):
        if len(obs.shape) == 1 or len(acs.shape) == 1:
            obs = np.squeeze(obs)[None]
            acs = np.squeeze(acs)[None]

        norm_obs = normalize(obs, data_statistics['obs_mean'], data_statistics['obs_std'])
        norm_acs = normalize(acs, data_statistics['acs_mean'], data_statistics['acs_std'])

        norm_input = torch.Tensor(np.concatenate((norm_obs, norm_acs), axis = 1)).to(self.device)
        norm_delta = self.delta_func(norm_input).cpu().detach().numpy()

        delta = unnormalize(norm_delta, data_statistics['delta_mean'], data_statistics['delta_std'])
        return obs + delta

    def update(self, observations, actions, next_observations, data_statistics):

        norm_obs = normalize(np.squeeze(observations), data_statistics['obs_mean'], data_statistics['obs_std'])
        norm_acs = normalize(np.squeeze(actions), data_statistics['acs_mean'], data_statistics['acs_std'])

        pred_delta = self.delta_func(torch.Tensor(np.concatenate((norm_obs, norm_acs), axis = 1)).to(self.device))
        true_delta = torch.Tensor(normalize(next_observations - observations, data_statistics['delta_mean'], data_statistics['delta_std'])).to(self.device)

        # TODO(Q1) Define a loss function that takes as input normalized versions of predicted change in state and true change in state
        loss = nn.functional.mse_loss(true_delta, pred_delta)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
