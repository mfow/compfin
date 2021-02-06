import pandas as pd
import numpy as np
from typing import List
from .column_types import CreditScoreCardColumnType


def internal_process_columns(**kwargs):
    data: pd.DataFrame = kwargs.get('data')
    assert isinstance(data, pd.DataFrame)

    predictors: List[str] = kwargs.get('predictors')
    response: str = kwargs.get('response')

    data_columns = list(data)

    if isinstance(response, int):
        response = data_columns[response]

    if predictors is None:
        predictors = list(data)
        predictors.remove(response)

    id_var: str = kwargs.get('id_var')
    if id_var is not None:
        if id_var in predictors:
            predictors.remove(id_var)

    assert response not in predictors

    for predictor in predictors:
        assert predictor in data_columns

    data_types: List[CreditScoreCardColumnType] = list()

    for predictor in predictors:
        predictor_series = data[predictor]
        if predictor_series.dtype in ['str', 'object']:
            data_type = CreditScoreCardColumnType.CATEGORICAL
        else:
            num_values = len(np.unique(predictor_series))

            if num_values > 10:
                data_type = CreditScoreCardColumnType.CONTINUOUS
            else:
                data_type = CreditScoreCardColumnType.CATEGORICAL

        data_types.append(data_type)

    if data is not None:
        data = data[[*predictors, response]].dropna()

    return {
        'data': data,
        'response': response,
        'predictors': predictors,
        'column_types': data_types
    }
