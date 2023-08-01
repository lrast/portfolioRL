import torch
import numpy as np


class AllocationMDP(object):
    def __init__(self, timeHorizon, mu, sigma):
        """
            The time units are set here so that each timestep is 1
            This of course impacts the 
        """
        self.timeHorizon = timeHorizon
        self.mu = mu
        self.sigma = sigma

    def updateTraces(self, newState, newAction):
        if self.stateTrace is None:
            self.stateTrace = newState[:, None, :].clone()
        else:
            self.stateTrace = torch.cat(
                                    [self.stateTrace, newState[:, None, :]],
                                    1)

        if self.actionTrace is None:
            self.actionTrace = newAction[:, None, :].clone()
        else:
            self.actionTrace = torch.cat(
                                    [self.actionTrace, newAction[:, None, :]],
                                    1)

    def initRun(self, Nsamples):
        """ Initialize a run """
        riskyAssets = torch.rand(Nsamples, 1)
        assets = torch.cat([1-riskyAssets, riskyAssets], 1)

        self.time = 0.
        self.state = torch.cat([assets, torch.zeros(Nsamples, 1)], 1)

        self.stateTrace = None
        self.actionTrace = None
        self.reward = torch.zeros(Nsamples, 1)

    def evolveState(self, actions):
        """
        action: the percent of the portfolio allocated to the risky asset
        """
        self.updateTraces(self.state, actions)

        totalAssets = self.state[:, 0:2].sum(1)

        self.time += 1
        self.state[:, 0] = (1.-actions[:, 0]) * totalAssets
        self.state[:, 1] = actions[:, 0] * totalAssets * \
            geometricBrownianMotion(self.mu, self.sigma, 1, 1, 
                                    reps=actions.shape[1])[:, 1]
        self.state[:, 2] = self.time * torch.ones(actions.shape[0])

        if self.time == self.timeHorizon:
            self.reward = self.state[:, 0:2].sum(1)[:, None]
            return True
        else:
            return False


def geometricBrownianMotion(mu, sigma, dt, N, reps=1):
    """ Generate geometric Brownian motion S:
        dS = mu S dt + sigma S dW
    """
    dW = dt**0.5 * np.random.randn(reps, N+1)
    dt = dt * np.ones((reps, N+1))

    dW[:, 0] = 0.0
    dt[:, 0] = 0.0

    W = np.cumsum(dW, axis=1)
    t = np.cumsum(dt, axis=1)

    return np.exp((mu - 0.5*sigma**2) * t + sigma * W)
