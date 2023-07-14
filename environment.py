import torch
import numpy as np


class allocationMDP(object):
    def __init__(self, timeHorizon, mu, sigma):
        """
            The time units are set here so that each timestep is 1
            This of course impacts the 
        """
        self.timeHorizon = timeHorizon
        self.mu = mu
        self.sigma = sigma

    def updateStateTrace(self, newState):
        self.stateTrace = torch.cat([self.stateTrace, newState[:, :, None]], 2)

    def initRun(self, Nsamples):
        """ Initialize a run """
        riskyAssets = torch.rand(Nsamples, 1)
        self.assets = torch.cat([1-riskyAssets, riskyAssets], 1)
        self.time = 0

        self.stateTrace = self.assets[:, :, None]
        self.reward = torch.zeros(Nsample, 1)

    def evolveState(self, actions):
        """
        action: the percent of the portfolio allocated to the risky asset
        """
        totalAssets = self.assets.sum(1)[:, None]

        self.assets = torch.cat([(1.-actions)*totalAssets, actions*totalAssets], 1)
        self.assets[:, 1] *= geometricBrownianMotion(self.mu, self.sigma, 1, 1, 
                                                     reps=actions.shape[1]
                                                     )[0, 1]

        self.updateStateTrace(self.assets)
        self.time += 1

        if self.time == self.timeHorizon:
            return True
            self.reward = self.assets.sum(1)[:, None]
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
