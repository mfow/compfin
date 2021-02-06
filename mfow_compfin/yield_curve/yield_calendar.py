from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List


class YieldCalendar(ABC):
    def __init__(self, **kwargs):
        # the epoch of the calendar.
        self.start_timestamp: datetime = kwargs.get('start_timestamp')
        assert isinstance(self.start_timestamp, datetime)

    @abstractmethod
    def get_period(self, timestamp: datetime) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_timestamp(self, period: int) -> datetime:
        raise NotImplementedError


class MonthlyYieldCalendar(YieldCalendar):
    def __init__(self, **kwargs):
        super(MonthlyYieldCalendar, self).__init__(**kwargs)
        assert self.start_timestamp.day == 1

    def get_timestamp(self, period: int) -> datetime:
        assert isinstance(period, int)
        assert period >= 0
        month = (self.start_timestamp.month + period - 1) % 12 + 1
        year = self.start_timestamp.year + int((self.start_timestamp.month + period - 1) / 12)
        return datetime(year=year, month=month, day=1)

    @staticmethod
    def __get_month_index(timestamp: datetime) -> int:
        return timestamp.month + timestamp.year * 12

    def get_period(self, timestamp: datetime) -> int:
        return self.__get_month_index(timestamp) - self.__get_month_index(self.start_timestamp)
