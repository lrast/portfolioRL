""" Tests for reasonable behavior of trained models. For different models, 
I expect different things. 
The state has three dimensions: risk-free and risky asset quantities, and time.

1. The decision made should not depend on the relative asset values for any model
2. The decision made should not depend on total for linear assets

"""

import pandas as pd
import torch

import statsmodels.api as sm
import statsmodels.formula.api as smf

from torch.distributions import Exponential


def generateData(model):
    times = torch.randint(0, 10, size=(1000, 1))
    totals = Exponential(1).sample((1000, 1))
    proportions = torch.rand((1000, 1))

    states = torch.cat([totals*proportions, totals*(1-proportions), times], 1)

    actionParams = model.forward(states)
    mean, var = model.actionStatistics(states)

    data = torch.cat([totals, proportions, times, actionParams, mean], 1)

    return pd.DataFrame(data.detach().numpy(), 
                        columns=['total', 'prop', 'time', 
                                 'theta1', 'theta2', 'meanAction'])


def runAnova(data, target='meanAction'):
    linModel = smf.ols(f'{target} ~ total + prop + time', data).fit()
    return sm.stats.anova_lm(linModel)
