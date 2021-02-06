import numpy as np


def sharpe_ratio(**kwargs) -> float:
    asset_returns: np.ndarray = kwargs.get('asset_returns')
    asset_return: float = kwargs.get('asset_return')
    asset_std_dev: float = kwargs.get('asset_std_dev')

    if asset_returns is None:
        assert asset_return is not None
        assert asset_std_dev is not None
    else:
        assert asset_return is None
        assert asset_std_dev is None

        if isinstance(asset_returns, list):
            asset_returns = np.array(asset_returns)

        assert len(asset_returns.shape) == 1
        asset_return = np.mean(asset_returns)
        asset_std_dev = np.std(asset_std_dev)

    assert asset_std_dev > 0.0

    risk_free_return: float = kwargs.get('risk_free_return')

    expected_excess = asset_return - risk_free_return
    return expected_excess / asset_std_dev

