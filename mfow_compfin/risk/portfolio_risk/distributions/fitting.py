import numpy as np
import scipy.stats as stats
from .distribution import Distribution
from .log_t_plus_risk import LogWithEntireInvestmentRiskDistribution


def fit_distribution(returns, **kwargs) -> Distribution:
    if isinstance(returns, list):
        returns = np.array(returns)

    assert len(returns.shape) == 1
    assert returns.shape[0] >= 5

    # remove nans
    returns = returns[np.logical_not(np.isnan(returns))]

    min_value = np.min(returns)

    # check that we never lose more than our entire investment
    assert min_value >= -1

    # compute the probability that we lose the entire investment.
    # this is modelled differently from the rest of the model.
    lose_entire_investment_pr = np.mean(returns == -1)

    # filter out the returns where we lose the entire investment.
    returns = returns[returns > -1]

    # transform the returns with log.
    returns = np.log(1 + returns)

    student_t_params = stats.t.fit(returns)
    student_t_model = stats.t(*student_t_params)

    return LogWithEntireInvestmentRiskDistribution(name=kwargs.get('name'),
                                                   model=student_t_model,
                                                   pr_lose_entire_investment=lose_entire_investment_pr)

