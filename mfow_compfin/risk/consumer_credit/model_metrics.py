import numpy as np
from sklearn.metrics import roc_auc_score



def get_model_metrics(**kwargs) -> dict:
    prediction: np.ndarray = kwargs.get('prediction')
    actual: np.ndarray = kwargs.get('actual')

    assert prediction.shape == actual.shape

    result = dict()

    result['accuracy'] = np.sum((prediction >= 0.5) == actual) / len(prediction)
    result['auroc'] = roc_auc_score(actual, prediction)

    return result

