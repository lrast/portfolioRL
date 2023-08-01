import torch

import pytorch_lightning as pl

from torch import nn

from torch.utils.data import DataLoader
from torch.distributions.beta import Beta

from environment import AllocationMDP


class PolicyLearning(pl.LightningModule):
    """agent that learn policy by the REINFORCE algorithm"""
    def __init__(self, **hyperparams):
        super(PolicyLearning, self).__init__()

        self.policyNet = nn.Sequential(
                                nn.Linear(3, 10),
                                nn.ReLU(),
                                nn.Linear(10, 10),
                                nn.ReLU(),
                                nn.Linear(10, 2),
                                SquareNlin()
                            )

        # default values
        hyperparameterValues = {
            'lr': 1e-3,
            'mu': 0.1,
            'sigma': 0.1,
            'timehorizon': 10,
            'batch_size': 500,
            'n_experiments': 10,
            'utilityFn': 'sqrt'
        }

        hyperparameterValues.update(hyperparams)
        self.save_hyperparameters(hyperparameterValues)

        # extract the utility function
        currentUtilityFn = self.hparams.utilityFn
        if isinstance(currentUtilityFn, str):
            self.utilityFn = {
                'linear': lambda x: x,
                'sqrt': lambda x: x**0.5,
                'log': lambda x: torch.log(x)
            }[currentUtilityFn]
        else:
            self.utilityFn = currentUtilityFn
            self.save_hyperparameters({'utilityFn': currentUtilityFn.__doc__})

    def forward(self, x):
        """ Sample actions from the policy """
        parameters = self.policyNet.forward(x)
        return parameters

    def sampleActions(self, state):
        parameters = self.forward(state)
        dists = Beta(parameters[:, 0], parameters[:, 1])

        return dists.sample()[:, None]

    def log_likelihood(self, states, actions):
        """ Returns the log-likelihood of given actions at given states
            The batching can have any shape, but we assume that 
        """
        parameters = self.forward(states)
        dists = Beta(parameters[..., 0:1], parameters[..., 1:2])
        return dists.log_prob(actions)

    def actionStatistics(self, states):
        parameters = self.forward(states)
        dists = Beta(parameters[..., 0:1], parameters[..., 1:2])

        return dists.mean, dists.variance

    def training_step(self, batch, batch_id):
        """Run a new epoch, apply the REINFORCE algorithm"""

        # run new epoch
        self.E.initRun(self.hparams.batch_size)

        stop = False
        while not stop:
            state = self.E.state
            actions = self.sampleActions(state)
            stop = self.E.evolveState(actions)

        loss = -self.utilityFn(self.E.reward[:, :, None]) * \
            self.log_likelihood(self.E.stateTrace, self.E.actionTrace)

        self.log('loss', loss.sum())
        self.log('mean action', 
                 self.actionStatistics(torch.tensor([0.5, 0.5, 0.])
                                       )[0].detach()
                 )
        return loss.sum()

    def setup(self, stage=None):
        """Initialize the environment"""
        self.E = AllocationMDP(self.hparams.timehorizon, self.hparams.mu,
                               self.hparams.sigma)

    def train_dataloader(self):
        """For now, we have no experience buffer, so the DL is empty"""
        return DataLoader(range(self.hparams.n_experiments))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class SquareNlin(nn.Module):
    """docstring for squareNlin"""
    def __init__(self):
        super(SquareNlin, self).__init__()

    def forward(self, x):
        return x**2 + 1E-16



# IN DEVELOPMENT
class Vlearner(pl.LightningModule):
    """value learning agent. IN DEVELOPMENT"""
    def __init__(self, utilityFn):
        super(Vlearner, self).__init__()

        self.utilityFn = utilityFn

        # the state is determined by two balances and the time
        self.VNetwork = nn.Sequential(
                                  nn.Linear(3, 10),
                                  nn.ReLU(),
                                  nn.Linear(10,10),
                                  nn.ReLU(),
                                  nn.Linear(10, 1),
                                  nn.Hardsigmoid()
                                )

    def Qvalues(self, states, actions):
        """Q value can be determined from the corresponding V values by changing the allocations
            while not evolving the time.
         """
        newAllocation = torch.stack([1-actions, actions])
        newAssets = (state[:, 0:2].sum(1) * newAllocation).T
        times = states[:, 2:3]

        return self.VNetwork(torch.cat([newAssets, times], 1))

    def optimalAction(self, state):
        pass

    def training_step(self, batch, batch_id):
        pass

    def forward(self, x):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
