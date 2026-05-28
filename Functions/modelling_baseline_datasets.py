import pandas as pd


def build_ar_features(logret: pd.Series, lags, ma_windows, vol_windows=(12, 36)) -> pd.DataFrame:
    feats = {}
    for lag in lags:
        feats[f'ar_ret_lag{lag}'] = logret.shift(lag)
    for window in ma_windows:
        feats[f'ar_ret_ma{window}'] = logret.shift(1).rolling(window).mean()
    for window in vol_windows:
        feats[f'ar_ret_std{window}'] = logret.shift(1).rolling(window).std()
    return pd.DataFrame(feats)


def register_dataset(
    registry,
    enabled,
    key,
    X_data,
    y_data,
    poly_cols_value=None,
    passthrough_cols_value=None,
):
    if not enabled:
        return
    registry[key] = {
        'X': X_data,
        'y': y_data,
        'poly_cols': list(poly_cols_value) if poly_cols_value is not None else None,
        'passthrough_cols': list(passthrough_cols_value) if passthrough_cols_value is not None else None,
    }
