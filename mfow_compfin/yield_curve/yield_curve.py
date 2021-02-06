import numpy as np
from typing import List
from functools import lru_cache
from .interest import interpolate_rate, discount_rate


class YieldCurve:
    def __init__(self, **kwargs):
        self.periods: np.ndarray = kwargs.get('periods')
        self.rates: np.ndarray = kwargs.get('rates')
        self.periods_per_year: int = kwargs.get('periods_per_year')

        if isinstance(self.periods, list):
            self.periods = np.array(self.periods)

        if isinstance(self.rates, list):
            self.rates = np.array(self.rates)

        self.allow_prior_extrapolation: bool = kwargs.get('allow_prior_extrapolation')
        self.allow_post_extrapolation: bool = kwargs.get('allow_post_extrapolation')
        self.allow_extrapolation: bool = kwargs.get('allow_extrapolation', True)

        if not self.allow_extrapolation:
            assert self.allow_prior_extrapolation is not True
            assert self.allow_post_extrapolation is not True
            self.allow_prior_extrapolation = False
            self.allow_post_extrapolation = False

        assert len(self.periods.shape) == 1
        assert len(self.rates.shape) == 1
        assert len(self.periods) == len(self.rates)
        assert len(self.periods) >= 1

        for i in range(1, len(self.periods)):
            assert self.periods[i] > self.periods[i-1]

        self.__period_lookup = dict()
        for i in range(len(self.periods)):
            self.__period_lookup[self.periods[i]] = i

    def __get_highest_prior_index(self, period: int, min: int, max: int) -> int:
        if min == max:
            return min

        assert min < max

        if min == max - 1:
            mid = min
        else:
            mid = int(np.floor((min + max) / 2))

        mid_value = self.periods[mid]
        if mid_value == period:
            return mid
        elif mid_value > period:
            return self.__get_highest_prior_index(period, min, mid - 1)
        else:
            if min == max - 1:
                if self.periods[max] <= period:
                    return max
                else:
                    return min

            return self.__get_highest_prior_index(period, mid, max)

    @lru_cache(maxsize=128)
    def __get_rate(self, period: int) -> float:
        if period < self.periods[0]:
            assert self.allow_prior_extrapolation
            return self.rates[0]
        elif period == self.periods[0]:
            return self.rates[0]

        prior_index = self.__get_highest_prior_index(period, 0, len(self.periods))
        prior_period = self.periods[prior_index]

        assert prior_period <= period

        if prior_period == period:
            return self.rates[prior_index]

        assert self.allow_extrapolation

        if prior_period == self.periods[-1]:
            assert self.allow_post_extrapolation
            next_index = prior_index
            prior_index = prior_index - 1
        else:
            next_index = prior_index + 1

        return interpolate_rate(first_period=self.periods[prior_index] / float(self.periods_per_year),
                                next_period=self.periods[next_index] / float(self.periods_per_year),
                                first_rate=self.rates[prior_index],
                                next_rate=self.rates[next_index],
                                extrapolate_period=period / float(self.periods_per_year))

    def get_rate(self, period):
        if period == 0:
            return 0.0

        if isinstance(period, int):
            return self.__get_rate(period)
        elif isinstance(period, list):
            return list(map(self.__get_rate, period))
        elif isinstance(period, np.ndarray):
            assert period.dtype == np.int
            assert len(period.shape) == 1
            return np.array(list(map(self.__get_rate, period)))

    def discount(self, start_period: int, end_period: int):
        start_discount = discount_rate(self.get_rate(start_period), start_period)
        end_discount = discount_rate(self.get_rate(end_period), end_period)
        assert start_discount > end_discount
        return end_discount / start_discount
