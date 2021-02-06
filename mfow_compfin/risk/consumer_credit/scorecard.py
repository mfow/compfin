import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from .consumer_columns import internal_process_columns
from .consumer_columns import CreditScoreCardColumnType
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from .model_metrics import get_model_metrics


class CreditScoreCard:
    def __init__(self, **kwargs):
        column_data = internal_process_columns(**kwargs)

        self.data: pd.DataFrame = column_data.get('data')
        self.response: str = column_data['response']
        self.predictors: List[str] = column_data['predictors']
        self.data_types: List[CreditScoreCardColumnType] = column_data['column_types']
        self.one_hot_encoders: Dict[str, OneHotEncoder] = dict()
        self.regression_model = LogisticRegression()

    def train_categories(self, data: Optional[pd.DataFrame] = None):
        if data is None:
            data = self.data

        assert isinstance(data, pd.DataFrame)

        for i in range(len(self.predictors)):
            if self.data_types[i] is CreditScoreCardColumnType.CATEGORICAL:
                predictor = self.predictors[i]
                encoder = OneHotEncoder()
                encoder.fit_transform(data[[predictor]])
                self.one_hot_encoders[predictor] = encoder

    def preprocess(self, data: Optional[pd.DataFrame] = None) -> np.ndarray:
        if data is None:
            data = self.data

        assert isinstance(data, pd.DataFrame)

        results: List[np.ndarray] = list()

        for i in range(len(self.predictors)):
            data_type = self.data_types[i]
            predictor = self.predictors[i]

            if data_type is CreditScoreCardColumnType.CONTINUOUS:
                result = data[[predictor]].to_numpy()
            elif data_type is CreditScoreCardColumnType.CATEGORICAL:
                result = self.one_hot_encoders[predictor].transform(data[[predictor]])
            elif data_type is CreditScoreCardColumnType.BIN:
                raise NotImplementedError()
            else:
                raise RuntimeError('Unsupported column type')

            results.append(result)

        x = np.concatenate(results, axis=1)
        y = np.array(data[self.response])
        return x, y

    def fit(self, data: Optional[pd.DataFrame] = None) -> np.ndarray:
        x, y = self.preprocess(data)
        self.regression_model.fit(x, y)

    def prob_default(self, data: Optional[pd.DataFrame] = None):
        x, y = self.preprocess(data)
        return self.regression_model.predict_proba(x)[:, 1]

    def evaluate(self, data: Optional[pd.DataFrame] = None) -> dict:
        x, y = self.preprocess(data)
        predicted = self.regression_model.predict_proba(x)[:, 1]
        return get_model_metrics(prediction=predicted, actual=y)
