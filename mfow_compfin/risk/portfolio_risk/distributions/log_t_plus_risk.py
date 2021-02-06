import numpy as np
import scipy.stats as stats
from abc import ABC, abstractmethod
from .distribution import Distribution


# a model for a distribution of returns
# where there is a fixed probability of losing an entire investment
# and for other cases, a log(students t) distribution
class LogWithEntireInvestmentRiskDistribution(Distribution):
    def __init__(self, **kwargs):
        self.model = kwargs.get('model')
        self.pr_lose_entire_investment = kwargs.get('pr_lose_entire_investment')

    def inv_cdf(self, p):
        y = p
        if isinstance(y, float):
            y = [y]

        if isinstance(y, list):
            y = np.array(y)

        assert isinstance(y, np.ndarray)
        assert len(y.shape) == 1
        assert np.min(y) >= -1
        assert np.max(y) <= 1

        result = self.model.ppf((y - self.pr_lose_entire_investment) / (1.0 - self.pr_lose_entire_investment))
        result = np.exp(result) - 1.0
        result[y <= self.pr_lose_entire_investment] = -1
        result[result < -1] = -1

        if isinstance(p, float):
            return result[0]

        return result

    def cdf(self, x):
        y = x
        if isinstance(y, float):
            y = [y]

        if isinstance(y, list):
            y = np.array(y)

        assert isinstance(y, np.ndarray)
        assert len(y.shape) == 1

        y_log = np.log(y + 1.0)
        pr = self.model.cdf(y_log)
        pr[y <= -1] = self.pr_lose_entire_investment

        if isinstance(x, float):
            return pr[0]

        return pr
