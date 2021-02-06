import numpy as np


def discount_rate(rate, periods):
    return 1.0 / np.power(1.0 + rate, periods)


def yield_rate(discount, periods):
    return np.power(1.0 / discount, 1.0 / periods) - 1.0


def interpolate_rate(**kwargs) -> float:
    first_period: int = kwargs.get('first_period')
    next_period: int = kwargs.get('next_period')
    extrapolate_period: int = kwargs.get('extrapolate_period')
    first_rate: float = kwargs.get('first_rate')
    next_rate: float = kwargs.get('next_rate')

    assert next_period > extrapolate_period > first_period

    if first_rate == next_rate:
        return first_rate

    first_discount = discount_rate(first_rate, first_period)
    next_discount = discount_rate(next_rate, next_period)
    assert 0.0 < next_discount < first_discount <= 1.0

    second_discount = next_discount / first_discount
    second_periods = next_period - first_period

    extrapolated_discount = first_discount * np.power(second_discount, (extrapolate_period - first_period) / second_periods)
    yr = yield_rate(extrapolated_discount, extrapolate_period)

    assert min(first_rate, next_rate) <= yr <= max(first_rate, next_rate)
    return yr
