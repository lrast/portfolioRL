import torch

import pytorch_lightning as pl

from torch import nn

from torch.utils.data import DataLoader
from torch.distributions import Beta, kl_divergence

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
            'regularization': 0,
            'baseline': 'none',

            'timehorizon': 10,
            'batch_size': 500,
            'n_experiments': 10,

            'mu': 0.1,
            'sigma': 0.1,
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
                'log': lambda x: torch.log(x), 
                '1000sqrt': lambda x: 1000*x**0.5
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

    def KLregularizer(self, states):
        """
            KL divergence regularizer
        """
        parameters = self.forward(states)
        currentParameters = parameters.detach()

        currentDist = Beta(currentParameters[..., 0:1], 
                           currentParameters[..., 1:2])
        newDist = Beta(parameters[..., 0:1], parameters[..., 0:1])

        return kl_divergence(currentDist, newDist)

    def actionStatistics(self, states):
        parameters = self.forward(states)
        dists = Beta(parameters[..., 0:1], parameters[..., 1:2])

        return dists.mean, dists.variance

    def training_step(self, batch, batch_id):
        """Run a new epoch, apply the REINFORCE algorithm"""

        batch_size_scheduler = self.hparams.batch_size
        if isinstance(batch_size_scheduler, int):
            batch_size = batch_size_scheduler
        else:
            batch_size = 200 + 10*batch_id

        # run new epoch
        self.E.initRun(batch_size)

        stop = False
        while not stop:
            state = self.E.state
            actions = self.sampleActions(state)
            stop = self.E.evolveState(actions)

        loss = -self.utilityFn(self.E.reward[:, :, None]) * \
            self.log_likelihood(self.E.stateTrace, self.E.actionTrace)

        if self.hparams.baseline == 'cashOut':
            # use as a baseline value function the value of cashing out now
            cashoutUtil = self.utilityFn(self.E.stateTrace[:, :, 0:2].sum(2))
            actualUtil = self.utilityFn(
                                        self.E.reward[:, :]
                                        ).repeat((1, self.hparams.timehorizon))
            loss = - (actualUtil - cashoutUtil)[:, :, None] * \
                self.log_likelihood(self.E.stateTrace, self.E.actionTrace)

        regularizer = self.hparams.regularization * \
            self.KLregularizer(self.E.stateTrace)

        self.log('loss', loss.mean())
        self.log('meanUtility', self.utilityFn(self.E.reward).mean())
        self.log('mean action', 
                 self.actionStatistics(torch.tensor([0.5, 0.5, 0.])
                                       )[0].detach()
                 )

        return (loss + regularizer).mean()

    def setup(self, stage=None):
        """Initialize the environment"""
        print(self.hparams)
        self.E = AllocationMDP(self.hparams.timehorizon, self.hparams.mu,
                               self.hparams.sigma)

    def train_dataloader(self):
        """For now, we have no experience buffer, so the DL is empty"""
        return DataLoader(range(self.hparams.n_experiments))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
        #return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class SquareNlin(nn.Module):
    """docstring for squareNlin"""
    def __init__(self):
        super(SquareNlin, self).__init__()

    def forward(self, x):
        return x**2 + 1E-16


class ConstantLearner(PolicyLearning):
    """Learns a constant policy output"""
    def __init__(self, **hyperparameters):
        super(ConstantLearner, self).__init__(**hyperparameters)

        self.policyNet = nn.Sequential(nn.Linear(3, 2),
                                       SquareNlin())
        self.policyNet[0].weight = nn.Parameter(torch.zeros(2, 3), 
                                                requires_grad=False)

    def validation_step(self, batch, batch_id):
        params = self.forward(torch.randn(1, 3))
        self.log('total certainty', params.sum())

    def val_dataloader(self):
        return DataLoader([0])
