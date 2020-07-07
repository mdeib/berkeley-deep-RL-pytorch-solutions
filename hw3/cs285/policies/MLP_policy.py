import numpy as np
import torch
import torch.nn as nn
from cs285.infrastructure.models import MLP

class MLPPolicy:
    def __init__(self,
        ac_dim,
        ob_dim,
        n_layers,
        size,
        device,
        learning_rate,
        training=True,
        discrete=False,
        nn_baseline=False,
        **kwargs):
        super().__init__()

        # init vars
        self.device = device
        self.discrete = discrete
        self.training = training
        self.nn_baseline = nn_baseline

        # network architecture
        self.policy_mlp = MLP(ac_dim, ob_dim, n_layers, size, device, discrete)
        params = list(self.policy_mlp.parameters())
        if self.nn_baseline:
            self.baseline_mlp = MLP(1, ob_dim, n_layers, size, device, True)
            params += list(self.baseline_mlp.parameters())

        #optimizer
        if self.training:
            self.optimizer = torch.optim.Adam(params, lr = learning_rate)

    ##################################

    # update/train this policy
    def update(self, observations, actions):
        raise NotImplementedError

    # query the neural net that's our 'policy' function, as defined by an mlp above
    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs):
        output = self.policy_mlp(torch.Tensor(obs).to(self.device))
        if self.discrete:
            action_probs = nn.functional.log_softmax(output).exp()
            return torch.multinomial(action_probs, num_samples = 1).cpu().detach().numpy()[0]
        else:
            return torch.normal(output[0], output[1]).cpu().detach().numpy()

    def get_log_prob(self, network_outputs, actions_taken):
        actions_taken = torch.Tensor(actions_taken).to(self.device)
        if self.discrete:
            network_outputs = nn.functional.log_softmax(network_outputs).exp()
            return torch.distributions.Categorical(network_outputs).log_prob(actions_taken)
        else:
            return torch.distributions.Normal(network_outputs[0], network_outputs[1]).log_prob(actions_taken).sum(-1)

#####################################################
#####################################################

class MLPPolicyPG(MLPPolicy):

    def update(self, observations, acs_na, adv_n = None, acs_labels_na = None, qvals = None):
        policy_output = self.policy_mlp(torch.Tensor(observations).to(self.device))
        logprob_pi = self.get_log_prob(policy_output, acs_na)

        self.optimizer.zero_grad()

        loss = torch.sum((-logprob_pi * torch.Tensor(adv_n).to(self.device)))
        loss.backward()

        if self.nn_baseline:
            baseline_prediction = self.baseline_mlp(torch.Tensor(observations).to(self.device)).view(-1)
            baseline_target = torch.Tensor((qvals - qvals.mean()) / (qvals.std() + 1e-8)).to(self.device)
            baseline_loss = nn.functional.mse_loss(baseline_prediction, baseline_target)
            baseline_loss.backward()

        self.optimizer.step()

        return loss

#####################################################
#####################################################

class MLPPolicyAC(MLPPolicyPG):
    """ MLP policy required for actor-critic.

    Note: Your code for this class could in fact the same as MLPPolicyPG, except the neural net baseline
    would not be required (i.e. self.nn_baseline would always be false. It is separated here only
    to avoid any unintended errors.
    """
    def __init__(self, *args, **kwargs):
        if 'nn_baseline' in kwargs.keys():
            assert kwargs['nn_baseline'] == False, "MLPPolicyAC should not use the nn_baseline flag"
        super().__init__(*args, **kwargs)
