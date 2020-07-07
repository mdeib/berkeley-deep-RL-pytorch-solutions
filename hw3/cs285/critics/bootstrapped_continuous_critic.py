import torch
from torch import nn
from cs285.infrastructure.models import MLP

class BootstrappedContinuousCritic:
    def __init__(self, hparams):
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.size = hparams['size']
        self.n_layers = hparams['n_layers']
        self.device = hparams['device']
        self.learning_rate = hparams['learning_rate']
        self.num_target_updates = hparams['num_target_updates']
        self.num_grad_steps_per_target_update = hparams['num_grad_steps_per_target_update']
        self.gamma = hparams['gamma']

        self.value_func = MLP(1, self.ob_dim, self.n_layers, self.size, self.device, True)
        # TODO: use the Adam optimizer to optimize the loss
        self.optimizer = torch.optim.Adam(self.value_func.parameters(), lr = self.learning_rate)

    def update(self, ob_no, next_ob_no, re_n, terminal_n):
        """
            Update the parameters of the critic.

            let sum_of_path_lengths be the sum of the lengths of the sampled paths
            let num_paths be the number of sampled paths

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                re_n: length: sum_of_path_lengths. Each element in re_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                loss
        """

        # TODO: Implement the pseudocode below:

        # do the following (self.num_grad_steps_per_target_update * self.num_target_updates) times:
            # every self.num_grad_steps_per_target_update steps (which includes the first step),
                # recompute the target values by
                    #a) calculating V(s') by querying this critic network (ie calling 'forward') with next_ob_no
                    #b) and computing the target values as r(s, a) + gamma * V(s')
                # HINT: don't forget to use terminal_n to cut off the V(s') (ie set it to 0) when a terminal state is reached
            # every time,
                # update this critic using the observations and targets
                # HINT: use nn.MSE()

        ob, next_ob, rew, done = map(lambda x: torch.from_numpy(x).to(self.device), [ob_no, next_ob_no, re_n, terminal_n])

        for update in range(self.num_grad_steps_per_target_update * self.num_target_updates):
            if update % self.num_grad_steps_per_target_update == 0:
                next_value = self.value_func(next_ob).squeeze() * (1 - done)
                target_value = rew + self.gamma * next_value

            self.optimizer.zero_grad()
            loss = nn.functional.mse_loss(self.value_func(ob).squeeze(), target_value)
            loss.backward()
            self.optimizer.step()
            target_value.detach_()

        return loss
