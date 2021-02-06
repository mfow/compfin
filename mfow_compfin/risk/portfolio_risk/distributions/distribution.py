import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List


class Distribution(ABC):
    def __init__(self, **kwargs):
        self.name: str = kwargs.get('name')

    @abstractmethod
    def cdf(self, x: Union[float, List[float], np.ndarray]):
        raise NotImplementedError

    @abstractmethod
    def inv_cdf(self, p: Union[float, List[float], np.ndarray]):
        raise NotImplementedError

    def value_at_risk(self, p: float = 0.05):
        assert 0.0 < p < 1.0
        return self.inv_cdf(p) * -1.0

    def conditional_value_at_risk(self, p: float = 0.05):
        assert 0.0 < p < 1.0
        pr = np.random.uniform(0.0, p, [10000])
        returns = self.inv_cdf(pr)
        return np.mean(returns) * -1.0


