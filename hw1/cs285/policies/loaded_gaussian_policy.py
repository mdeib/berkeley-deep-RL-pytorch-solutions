import numpy as np
import torch
import torch.nn as nn
import pickle

class Loaded_Gaussian_Policy(nn.Module):
    def __init__(self, filename, **kwargs):
        super().__init__()
        with open(filename, 'rb') as f:
            data = pickle.loads(f.read())

        self.nonlin_type = data['nonlin_type']
        policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]

        assert policy_type == 'GaussianPolicy', 'Policy type {} not supported'.format(policy_type)
        self.policy_params = data[policy_type]

        assert set(self.policy_params.keys()) == {'logstdevs_1_Da', 'hidden', 'obsnorm', 'out'}

        self.obsnorm_mean = self.policy_params['obsnorm']['Standardizer']['mean_1_D']
        self.obsnorm_meansq = self.policy_params['obsnorm']['Standardizer']['meansq_1_D']
        layer_params = self.policy_params['hidden']['FeedforwardNet']

        self.mlp = nn.ModuleList()
        for layer_name in sorted(layer_params.keys()):
            W = layer_params[layer_name]['AffineLayer']['W'].astype(np.float32)
            b = layer_params[layer_name]['AffineLayer']['b'].astype(np.float32)
            r, h = W.shape

            layer = nn.Linear(r,h)
            layer.weight.data.copy_(torch.from_numpy(W.transpose()))
            layer.bias.data.copy_(torch.from_numpy(b.squeeze(0)))
            self.mlp.append(layer)

            if self.nonlin_type == 'lrelu':
                self.mlp.append(nn.LeakyReLU())
            elif self.nonlin_type == 'tanh':
                self.mlp.append(nn.Tanh())
            else:
                raise NotImplementedError(self.nonlin_type)

        #output layer
        W = self.policy_params['out']['AffineLayer']['W'].astype(np.float32)
        b = self.policy_params['out']['AffineLayer']['b'].astype(np.float32)
        r, h = W.shape
        layer = nn.Linear(r, h)
        layer.weight.data.copy_(torch.from_numpy(W.transpose()))
        layer.bias.data.copy_(torch.from_numpy(b.squeeze(0)))
        self.mlp.append(layer)

    ##################################

    def obs_norm(self, obs_bo, obsnorm_mean, obsnorm_meansq):
        obsnorm_stdev = np.sqrt(np.maximum(0, obsnorm_meansq - np.square(obsnorm_mean)))
        normedobs_bo = (obs_bo - obsnorm_mean) / (obsnorm_stdev + 1e-6)
        return torch.FloatTensor(normedobs_bo).squeeze(0)

    ##################################

    def forward(self, obs):
        x = self.obs_norm(obs, self.obsnorm_mean, self.obsnorm_meansq)
        for layer in self.mlp:
            x = layer(x)
        return x

    ##################################

    def update(self, obs_no, acs_na, adv_n=None, acs_labels_na=None):
        print("\n\nThis policy class simply loads in a particular type of policy and queries it.")
        print("Not training procedure has been written, so do not try to train it.\n\n")
        raise NotImplementedError

    def get_action(self, obs):
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None, :]
        return self(obs)
