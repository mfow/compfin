import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from .consumer_columns import internal_process_columns
from .column_types import CreditScoreCardColumnType
from .model_metrics import get_model_metrics


def __screen_predictor(predictor: pd.Series, response: pd.Series, **kwargs):
    assert len(predictor) == len(response)
    original_length = len(predictor)

    df = pd.DataFrame({
        'predictor': predictor,
        'response': response
    }).dropna()

    result = dict()

    result['percent_missing'] = 100.0 - (len(df) * 100.0 / original_length)

    column_type: CreditScoreCardColumnType = kwargs.get('column_type')

    if column_type is CreditScoreCardColumnType.CATEGORICAL:
        encoder = OneHotEncoder()
        predictor_np = encoder.fit_transform(df[['predictor']])
    elif column_type is CreditScoreCardColumnType.CONTINUOUS:
        predictor_np = np.expand_dims(df['predictor'].to_numpy(), axis=1)
    elif column_type is CreditScoreCardColumnType.BIN:
        raise RuntimeError('not implemented')
    else:
        raise RuntimeError('Not supported column type')

    response_np = df['response'].to_numpy()

    reg_model = LogisticRegression()
    reg_model.fit(predictor_np, response_np)
    predicted = reg_model.predict_proba(predictor_np)[:, 1]

    metrics = get_model_metrics(prediction=predicted, actual=response_np)
    for metric in metrics.keys():
        result[metric] = metrics[metric]

    return result


def screen_predictors(data: pd.DataFrame, **kwargs):
    column_data = internal_process_columns(data=data, **kwargs)

    predictors = column_data['predictors']
    response = column_data['response']
    column_types = column_data['column_types']

    rows: List[dict] = list()
    response_series = data[response]

    for i in range(len(predictors)):
        predictor = predictors[i]
        row = __screen_predictor(data[predictor], response_series, column_type=column_types[i])
        row['predictor'] = predictor
        rows.append(row)

    results = pd.DataFrame(rows)

    return results.set_index('predictor')
