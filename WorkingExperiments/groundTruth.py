import numpy as np

from scipy.optimize import minimize, LinearConstraint
from scipy.stats import lognorm
from scipy.integrate import quad

from environment import geometricBrownianMotion


def findBestAction(mu, sigma, initialVal, utilityFn):
    """Optimize the utility function by simple gradient ascent"""
    samples = geometricBrownianMotion(mu, sigma, 1, 1, reps=500000)[:, 1]

    def objectiveFn(weight):
        return -utilityFn((1-weight)*initialVal + 
                          weight*initialVal*samples).sum()

    constraints = LinearConstraint(np.array(1), lb=0., ub=1.)

    return minimize(objectiveFn, 0.5, constraints=constraints)


def findBestAction_determ(mu, sigma, initialVal, utilityFn):
    """Deterministic version using numerical quadrature"""
    # geometric brownian motion is log normal with variance corrected mean

    geoBMpdf = lognorm(sigma, scale=np.exp(mu-sigma**2/2)).pdf

    def analyticObjective(weight):
        """Analytic integral in the objective function"""
        def toIntegrate(x):
            """body of the integral"""
            utilityVal = utilityFn((1-weight)*initialVal +
                                   weight*initialVal*x)
            return utilityVal * geoBMpdf(x)

        return -quad(toIntegrate, 0., np.inf)[0]  # negative to minimize

    constraints = LinearConstraint(np.array(1), lb=0., ub=1.)

    return minimize(np.vectorize(analyticObjective), 0.5,
                    constraints=constraints)


def utilityCurve_determ(mu, sigma, initialVal, utilityFn):
    """Full utility curve, no maximization"""
    geoBMpdf = lognorm(sigma, scale=np.exp(mu-sigma**2/2)).pdf

    def analyticObjective(weight):
        """Analytic integral in the objective function"""
        def toIntegrate(x):
            """body of the integral"""
            utilityVal = utilityFn((1-weight)*initialVal +
                                   weight*initialVal*x)
            return utilityVal * geoBMpdf(x)

        return quad(toIntegrate, 0., np.inf)[0]  # negative to minimize

    return np.vectorize(analyticObjective)
