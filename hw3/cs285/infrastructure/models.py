import torch
from torch import nn

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

class LL_DQN(MLP):
    def __init__(self, ac_dim, ob_dim, device):
        super().__init__(ac_dim, ob_dim, 2, 64, device, True, nn.ReLU())

class atari_DQN(nn.Module):
    def __init__(self, ac_dim, ob_dim, device):
        super().__init__()

        self.convnet = nn.Sequential(
            nn.Conv2d(ob_dim[2], 32, 8, stride = 4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride = 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride = 1),
            nn.ReLU(True),
        )
        self.action_value = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(True),
            nn.Linear(512, ac_dim),
        )
        self.to(device)

    def forward(self, obs):
        out = obs.float() / 255
        out = out.permute(0, 3, 1, 2) #reshape to [batch size, channels, height, width]
        out = self.convnet(out)
        out = out.reshape(out.size(0), -1)
        out = self.action_value(out)
        return out

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def restore(self, filepath):
        self.load_state_dict(torch.load(filepath))
