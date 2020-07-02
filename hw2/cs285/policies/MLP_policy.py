import numpy as np
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,
        ac_dim,
        ob_dim,
        n_layers,
        size,
        device,
        discrete,
        activation = nn.Tanh()):
        super().__init__()

        self.discrete = discrete

        # network architecture
        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(ob_dim, size)) #first hidden layer
        self.mlp.append(activation)

        for h in range(n_layers - 1): #additional hidden layers
            self.mlp.append(nn.Linear(size, size))
            self.mlp.append(activation)

        self.mlp.append(nn.Linear(size, ac_dim)) #output layer, no activation function

        #if continuous define logstd variable
        if not self.discrete:
            self.logstd = nn.Parameter(torch.zeros(ac_dim))

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
            #baseline_loss = nn.functional.mse_loss(baseline_prediction, torch.Tensor(qvals).to(self.device))
            baseline_loss.backward()

        self.optimizer.step()

        return loss

#####################################################
#####################################################
