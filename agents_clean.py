import torch

import pytorch_lightning as pl

from torch import nn

from torch.utils.data import DataLoader
from torch.distributions import Beta
from torch.special import polygamma

from environment import AllocationMDP


class BetaDistributionLearner(pl.LightningModule):
    """agent that learn policy by the REINFORCE algorithm"""
    def __init__(self, **hyperparams):
        super(BetaDistributionLearner, self).__init__()

        self.policyNet = nn.Sequential(
                                nn.Embedding(10, 6),
                                nn.Linear(6, 2),
                                AbsNlin()
                                )

        # default values
        hyperparameterValues = {
            'lr': 1e-3,
            'baseline': 'cashout',

            'timehorizon': 10,
            'batch_size': 1000,
            'n_experiments': 100,

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
                'sqrt': lambda x: x**0.5,
                'log': lambda x: torch.log(x), 
            }[currentUtilityFn]
        else:
            self.utilityFn = currentUtilityFn
            self.save_hyperparameters({'utilityFn': currentUtilityFn.__doc__})

    """
        Forward operation functions
    """

    def forward(self, states):
        times = states[..., 2].type(torch.int)
        return self.policyNet(times)

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
        return dists.log_prob(actions), parameters

    def actionStatistics(self, states):
        parameters = self.forward(states)
        dists = Beta(parameters[..., 0:1], parameters[..., 1:2])

        return dists.mean, dists.variance

    def InverseFisherInfo(self, parameters):
        with torch.no_grad():
            a = parameters[..., 0:1]
            b = parameters[..., 1:2]
            M11 = polygamma(1, a) - polygamma(1, a+b)
            M12 = - polygamma(1, a+b)
            M21 = - polygamma(1, a+b)
            M22 = polygamma(1, b) - polygamma(1, a+b)

            invDeterm = (1. / (M11 * M22 - M12 * M21))[..., None]
            invFisherMatrices = invDeterm * torch.cat(
                    [torch.cat([M22, -M12], dim=-1)[..., None],
                     torch.cat([-M21, M22], dim=-1)[..., None]],
                    dim=-1
                )
        return invFisherMatrices

    """
        Training 
    """

    def training_step(self, batch, batch_id):
        """Run a new epoch, apply the REINFORCE algorithm"""

        # run new epoch
        self.E.initRun(self.hparams.batch_size)

        stop = False
        while not stop:
            state = self.E.state
            actions = self.sampleActions(state)
            stop = self.E.evolveState(actions)

        observedUtility = self.utilityFn(
                                    self.E.reward[:, :]
                                    ).repeat((1, self.hparams.timehorizon))

        if self.hparams.baseline == 'cashout':
            # use as a baseline value function the value of cashing out now
            baseline = self.utilityFn(self.E.stateTrace[:, :, 0:2].sum(2))
        else:
            baseline = 0.

        logProb, parameters = self.log_likelihood(self.E.stateTrace, 
                                                  self.E.actionTrace)
        loss = - (observedUtility - baseline)[:, :, None] * logProb

        # make Fisher Information matrices for use in natural gradient descent
        self.currentParameters = parameters
        self.currentParameters.retain_grad()

        self.log('loss', loss.mean())
        self.log('meanUtility', self.utilityFn(self.E.reward).mean())
        self.log('mean action', 
                 self.actionStatistics(torch.tensor([0.5, 0.5, 0.])
                                       )[0].detach()
                 )

        return loss.mean()

    def backward(self, loss, **kwargs):
        loss.backward(retain_graph=True)
        parameterGrads = self.currentParameters.grad
        InverseFisher = self.InverseFisherInfo(self.currentParameters)

        naturalGradients = torch.einsum('nmij, nmj -> nmi', 
                                        InverseFisher, parameterGrads)

        self.policyNet.zero_grad()
        torch.autograd.backward(self.currentParameters, grad_tensors=naturalGradients)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    """
        Data handling
    """

    def setup(self, stage=None):
        """Initialize the environment"""
        self.E = AllocationMDP(self.hparams.timehorizon, self.hparams.mu,
                               self.hparams.sigma)

    def train_dataloader(self):
        """For now, we have no experience buffer, so the DL is empty"""
        return DataLoader(range(self.hparams.n_experiments))

    def validation_step(self, batch, batch_id):
        params = self.forward(torch.arange(0, 10)[:, None].repeat(1, 3))
        self.log('total certainty', params.sum(1).mean())

    def val_dataloader(self):
        return DataLoader([0])


"""

            Utility: nonlinearities as layers

"""


class AbsNlin(nn.Module):
    """docstring for squareNlin"""
    def __init__(self):
        super(AbsNlin, self).__init__()

    def forward(self, x):
        return torch.abs(x)
