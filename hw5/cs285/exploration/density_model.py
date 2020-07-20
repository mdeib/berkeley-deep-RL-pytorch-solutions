import numpy as np
import torch
from torch import nn
from cs285.infrastructure.utils import MLP

class Histogram:
    def __init__(self, nbins, preprocessor):
        self.nbins = nbins
        self.total = 0.
        self.hist = {}
        for i in range(int(self.nbins)):
            self.hist[i] = 0
        self.preprocessor = preprocessor

    def update_count(self, state, increment):
        """
            ### PROBLEM 1
            ### YOUR CODE HERE

            args:
                state: numpy array
                increment: int

            TODO:
                1. increment the entry "bin_name" in self.hist by "increment"
                2. increment self.total by "increment"
        """
        bin_name = self.preprocessor(state)
        self.hist[bin_name] += increment
        self.total += increment

    def get_count(self, states):
        """
            ### PROBLEM 1
            ### YOUR CODE HERE

            args:
                states: numpy array (bsize, ob_dim)

            returns:
                counts: numpy_array (bsize)

            TODO:
                For each state in states:
                    1. get the bin_name using self.preprocessor
                    2. get the value of self.hist with key bin_name
        """
        counts = []
        for state in states:
            bin_name = self.preprocessor(state)
            counts.append(self.hist[bin_name])
        return np.array(counts)

    def get_prob(self, states):
        """
            ### PROBLEM 1
            ### YOUR CODE HERE

            args:
                states: numpy array (bsize, ob_dim)

            returns:
                return the probabilities of the state (bsize)

            NOTE:
                remember to normalize by float(self.total)
        """
        probs = self.get_count(states) / self.total
        return probs

class RBF:
    """
        https://en.wikipedia.org/wiki/Radial_basis_function_kernel
        https://en.wikipedia.org/wiki/Kernel_density_estimation
    """
    def __init__(self, sigma):
        self.sigma = sigma
        self.means = None

    def fit_data(self, data):
        """
            ### PROBLEM 2
            ### YOUR CODE HERE

            args:
                data: list of states of shape (ob_dim)

            TODO:
                We simply assign self.means to be equal to the data points.
                Let the length of the data be B
                self.means: np array (B, ob_dim)
        """
        B, ob_dim = len(data), len(data[0])
        self.means = np.array(data)
        assert self.means.shape == (B, ob_dim)

    def get_prob(self, states):
        """
            ### PROBLEM 2
            ### YOUR CODE HERE

            given:
                states: (b, ob_dim)
                    where b is the number of states we wish to get the
                    probability of

                self.means: (B, ob_dim)
                    where B is the number of states in the replay buffer
                    we will plop a Gaussian distribution on top of each
                    of self.means with a std of self.sigma

            TODO:
                1. Compute deltas: for each state in states, compute the
                    difference between that state and every mean in self.means.
                2. Euclidean distance: sum the squared deltas
                3. Gaussian: evaluate the probability of the state under the
                    gaussian centered around each mean. The hyperparameters
                    for the reference solution assume that you do not normalize
                    the gaussian. This is fine since the rewards will be
                    normalized later when we compute advantages anyways.
                4. Average: average the probabilities from each gaussian
        """
        b, ob_dim = states.shape
        if self.means is None:
            # Return a uniform distribution if we don't have samples in the
            # replay buffer yet.
            return (1.0/len(states))*np.ones(len(states))
        else:
            B, replay_dim = self.means.shape
            assert states.ndim == self.means.ndim and ob_dim == replay_dim

            # 1. Compute deltas
            deltas = np.array([state - self.means for state in states])
            assert deltas.shape == (b, B, ob_dim)

            # 2. Euclidean distance
            euc_dists = np.sum(np.square(deltas), axis = 2)
            assert euc_dists.shape == (b, B)

            # Gaussian
            gaussians = np.exp(-euc_dists / (2 * self.sigma ** 2))
            assert gaussians.shape == (b, B)

            # 4. Average
            densities = np.mean(gaussians, axis = 1)
            assert densities.shape == (b,)

            return densities

class Exemplar(nn.Module):
    def __init__(self, ob_dim, hid_dim, learning_rate, kl_weight, device):
        super().__init__()
        self.ob_dim = ob_dim
        self.hid_dim = hid_dim
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.device = device

        self.encoder1 = MLP(input_dim = self.ob_dim,
                            output_dim = self.hid_dim // 2,
                            n_layers = 2,
                            size = self.hid_dim,
                            device = self.device,
                            discrete = False)

        self.encoder2 = MLP(input_dim = self.ob_dim,
                            output_dim = self.hid_dim // 2,
                            n_layers = 2,
                            size = self.hid_dim,
                            device = self.device,
                            discrete = False)

        self.discriminator = MLP(input_dim = self.hid_dim,
                                output_dim = 1,
                                n_layers = 2,
                                size = self.hid_dim,
                                device = self.device,
                                discrete = True)

        prior_means = torch.zeros(self.hid_dim // 2).to(self.device)
        prior_cov = torch.eye(self.hid_dim // 2).to(self.device)
        self.prior = torch.distributions.MultivariateNormal(prior_means, prior_cov)

        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)

    def forward(self, state1, state2):
        encoded1_mean, encoded1_std = self.encoder1(torch.Tensor(state1).to(self.device))
        encoded2_mean, encoded2_std = self.encoder2(torch.Tensor(state2).to(self.device))

        epsilon1 = self.prior.sample().to(self.device)
        epsilon2 = self.prior.sample().to(self.device)

        #Reparameterization trick
        latent1 = encoded1_mean + (encoded1_std * epsilon1)
        latent2 = encoded2_mean + (encoded2_std * epsilon2)

        logit = self.discriminator(torch.cat([latent1, latent2], axis = 1)).squeeze()
        return logit

    def get_log_likelihood(self, state1, state2, target):
        """
            ### PROBLEM 3
            ### YOUR CODE HERE

            args:
                state1: np array (batch_size, ob_dim)
                state2: np array (batch_size, ob_dim)

            TODO:
                likelihood of state1 == state2

            Hint:
                what should be the value of self.discrim_target?
        """
        logit = self(state1, state2)
        discriminator_dist = torch.distributions.Bernoulli(logit)
        log_likelihood = discriminator_dist.log_prob(torch.Tensor(target).to(self.device).squeeze())
        return log_likelihood

    def update(self, state1, state2, target):
        """
            ### PROBLEM 3
            ### YOUR CODE HERE

            args:
                state1: np array (batch_size, ob_dim)
                state2: np array (batch_size, ob_dim)
                target: np array (batch_size, 1)

            TODO:
                train the density model and return
                    ll: log_likelihood
                    kl: kl divergence
                    elbo: elbo
        """
        assert state1.ndim == state2.ndim == target.ndim
        assert state1.shape[1] == state2.shape[1] == self.ob_dim
        assert state1.shape[0] == state2.shape[0] == target.shape[0]

        log_likelihood = self.get_log_likelihood(state1, state2, target)

        encoded1_mean, encoded1_std = self.encoder1(torch.Tensor(state1).to(self.device))
        encoded2_mean, encoded2_std = self.encoder2(torch.Tensor(state2).to(self.device))

        encoded1_dist = torch.distributions.MultivariateNormal(encoded1_mean, torch.diag(encoded1_std ** 2))
        encoded2_dist = torch.distributions.MultivariateNormal(encoded2_mean, torch.diag(encoded2_std ** 2))

        kl1 = torch.distributions.kl.kl_divergence(encoded1_dist, self.prior)
        kl2 = torch.distributions.kl.kl_divergence(encoded2_dist, self.prior)

        kl = (kl1 + kl2)
        elbo = (log_likelihood - (kl * self.kl_weight)).mean()

        self.optimizer.zero_grad()
        (-elbo).backward()
        self.optimizer.step()

        ll, kl, elbo = map(lambda x: x.cpu().detach().numpy(), (log_likelihood, kl, elbo))
        return ll, kl, elbo

    def get_prob(self, state):
        """
            ### PROBLEM 3
            ### YOUR CODE HERE

            args:
                state: np array (batch_size, ob_dim)

            TODO:
                likelihood:
                    evaluate the discriminator D(x,x) on the same input
                prob:
                    compute the probability density of x from the discriminator
                    likelihood (see homework doc)
        """
        #since liklihood of target 1 is just the value of the logit
        likelihood = self(state, state).cpu().detach().numpy()

        # avoid divide by 0 and log(0)
        likelihood = np.clip(np.squeeze(likelihood), 1e-5, 1-1e-5)
        prob = (1 - likelihood) / likelihood
        return prob
