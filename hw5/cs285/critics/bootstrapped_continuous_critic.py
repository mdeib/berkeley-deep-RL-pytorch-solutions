import torch
from torch import nn
from cs285.infrastructure.utils import MLP

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

        self.value_func = MLP(self.ob_dim, 1, self.n_layers, self.size, self.device, True)
        self.optimizer = torch.optim.Adam(self.value_func.parameters(), lr = self.learning_rate)

    def update(self, ob_no, next_ob_no, re_n, terminal_n):
        '''
        ts_ob_no, ts_next_ob_no, ts_re_n, ts_terminal_n = map(lambda x: torch.Tensor(x).to(self.device),
                                                              [ob_no, next_ob_no, re_n, terminal_n])
        for _ in range(self.num_target_updates):
            with torch.no_grad():
                ts_next_V_n = self.value_func(ts_next_ob_no).view(-1)
            ts_target_n = ts_re_n + (1 - ts_terminal_n) * self.gamma * ts_next_V_n
            for _ in range(self.num_grad_steps_per_target_update):
                ts_V_n = self.value_func(ts_ob_no).view(-1)
                self.optimizer.zero_grad()
                loss = nn.functional.mse_loss(ts_V_n, ts_target_n)
                loss.backward()
                self.optimizer.step()
        '''
        ob, next_ob, rew, done = map(lambda x: torch.Tensor(x).to(self.device), [ob_no, next_ob_no, re_n, terminal_n])

        for update in range(self.num_grad_steps_per_target_update * self.num_target_updates):
            if update % self.num_grad_steps_per_target_update == 0:
                next_value = self.value_func(next_ob).squeeze() * (1 - done)
                target_value = rew + self.gamma * next_value

            self.optimizer.zero_grad()
            loss = nn.functional.mse_loss(self.value_func(ob).squeeze(), target_value)
            loss.backward()
            self.optimizer.step()
            target_value.detach_()
        #'''

        return loss
