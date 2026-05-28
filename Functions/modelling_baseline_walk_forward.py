import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def walk_forward_indices(n: int, kind: str, size: int, step: int = 1):
    assert kind in ('fixed', 'expanding')
    for idx in range(size, n, step):
        train_slice = slice(idx - size, idx) if kind == 'fixed' else slice(0, idx)
        test_slice = slice(idx, min(idx + step, n))
        if test_slice.stop <= test_slice.start:
            break
        yield train_slice, test_slice


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        'n': int(len(y_true)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float('nan'),
        'dir_acc': float(np.mean(np.sign(y_true) == np.sign(y_pred))),
    }


def walk_forward(
    model_name: str,
    scheme_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    model_factory,
    *,
    window_schemes: dict,
    lstm_test_block: int,
):
    kind, size = window_schemes[scheme_name]
    step = lstm_test_block if model_name.startswith('lstm') else 1
    y_true, y_pred, ts_pred = [], [], []
    train_time_s = 0.0
    iterations = list(walk_forward_indices(len(X), kind, size, step=step))

    for idx, (train_slice, test_slice) in enumerate(iterations):
        model = model_factory()
        if model is None:
            return None
        start = time.perf_counter()
        model.fit(X.iloc[train_slice], y.iloc[train_slice])
        train_time_s += time.perf_counter() - start
        pred = model.predict(X.iloc[test_slice])
        y_true.extend(y.iloc[test_slice].tolist())
        y_pred.extend(np.asarray(pred).tolist())
        ts_pred.extend(y.iloc[test_slice].index.tolist())
        if idx % 50 == 0:
            print(
                f'  [{model_name}/{scheme_name}] step {idx + 1}/{len(iterations)} '
                f'(train={train_slice.stop - train_slice.start}, test={test_slice.stop - test_slice.start})'
            )

    return {
        'y_true': np.asarray(y_true),
        'y_pred': np.asarray(y_pred),
        'timestamps': ts_pred,
        'final_model': model,
        'train_time_s': train_time_s,
    }
