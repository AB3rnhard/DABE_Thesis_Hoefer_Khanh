import datetime as dt
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def file_sha1(path: Path, bufsize=1 << 20) -> str:
    hasher = hashlib.sha1()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(bufsize), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def artefact_paths(model_name: str, scheme: str, data_date: str, n_features: int, cfg: dict, dataset_tag: str = 'poly') -> tuple[Path, Path]:
    tokens = [
        model_name.upper(),
        scheme,
        dataset_tag,
        f"h{cfg['return_horizon_min']}m",
        f'f{n_features}',
        data_date,
    ]
    stem = '_'.join(tokens)
    ext = '.keras' if model_name.startswith('lstm') else '.pkl'
    return cfg['models_dir'] / f'{stem}{ext}', cfg['results_dir'] / f'{stem}.json'


def cached_run_valid(meta_path: Path, data_hash: str, force_retrain: bool) -> bool:
    if not meta_path.exists():
        return False
    pred_path = meta_path.with_name(meta_path.stem + '.predictions.csv')
    if not pred_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False
    return meta.get('data_sha1') == data_hash and not force_retrain


def save_artefacts(
    model_name,
    scheme,
    data_date,
    data_hash,
    metrics,
    result,
    X_cols,
    *,
    cfg: dict,
    train_time_s=None,
    dataset_tag='poly',
):
    n_features = len(X_cols)
    model_path, meta_path = artefact_paths(model_name, scheme, data_date, n_features, cfg, dataset_tag=dataset_tag)
    model = result['final_model']

    if model_name.startswith('lstm'):
        if model_name == 'lstm':
            core = model
            transformer = None
        else:
            core = model.core
            transformer = model.transformer
        core.model.save(model_path)
        state = {
            'xs': core.xs,
            'ys': core.ys,
            'seq_len': core.seq_len,
            'last_train_X': core._last_train_X,
        }
        if transformer is not None:
            state['transformer'] = transformer
        joblib.dump(state, model_path.with_suffix('.scalers.pkl'))
    else:
        joblib.dump(model, model_path)

    uses_pls = 'pls' in model_name
    trad_dataset = dataset_tag in {'trad', 'poly+trad', 'bloomberg_only'}
    meta = {
        'model': model_name,
        'window': scheme,
        'dataset_tag': dataset_tag,
        'data_date': data_date,
        'data_sha1': data_hash,
        'run_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'train_time_s': round(train_time_s, 3) if train_time_s is not None else None,
        'n_features': n_features,
        'feature_sample': X_cols[:15],
        'metrics': metrics,
        'config': {
            'bar_minutes': cfg['bar_minutes'],
            'return_horizon_min': cfg['return_horizon_min'],
            'horizon_steps': cfg['horizon_steps'],
            'ar_lags': cfg['ar_lags'],
            'ar_ma_windows': cfg['ar_ma_windows'],
            'feature_engineering_done': cfg['feature_engineering_done'],
            'prefilter_topN': cfg['max_features_prefilter'] if not cfg['feature_engineering_done'] else None,
            'lstm_seq_len': cfg['lstm_seq_len'],
            'lstm_block': cfg['lstm_test_block'],
            'lstm_epochs': cfg['lstm_epochs'],
            'lstm_batch': cfg['lstm_batch'],
            'rf_estimators': cfg['rf_n_estimators'],
            'random_state': cfg['random_state'],
            'handle_daily_gap': cfg['handle_daily_gap'],
            'gap_threshold': cfg['gap_threshold'],
            'fe_input_mode': cfg['fe_input_mode'],
            'trad_max_ffill_bars': cfg['traditional_max_ffill_bars'] if trad_dataset else None,
            'trad_use_staleness': cfg['traditional_use_staleness'] if trad_dataset else None,
            'trad_ar_lags': cfg['traditional_ar_lags'] if trad_dataset else None,
            'pls_n_components': cfg['pls_n_components'] if uses_pls else None,
            'pls_selection_path': str(cfg['pls_selection_artefact_path']) if uses_pls else None,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    pred_path = meta_path.with_name(meta_path.stem + '.predictions.csv')
    assert len(result['timestamps']) == len(result['y_true']) == len(result['y_pred'])
    pd.DataFrame({
        'timestamp': result['timestamps'],
        'y_true': np.asarray(result['y_true']),
        'y_pred': np.asarray(result['y_pred']),
    }).to_csv(pred_path, index=False)

    train_time_str = f'{train_time_s:.1f}s' if train_time_s is not None else 'N/A'
    with open(cfg['results_dir'] / 'runs_log.txt', 'a', encoding='utf-8') as handle:
        handle.write(
            f"{meta['run_utc']}  {model_name:9s}  {scheme:10s}  {dataset_tag:14s}  "
            f"h={cfg['return_horizon_min']}m  f={n_features}  data={data_date}  "
            f"rmse={metrics['rmse']:.5e}  mae={metrics['mae']:.5e}  "
            f"r2={metrics['r2']:+.4f}  dir_acc={metrics['dir_acc']:.3f}  "
            f"train_time={train_time_str}\n"
        )
    with open(cfg['results_dir'] / 'runs_log.jsonl', 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(meta, default=str) + '\n')

    return meta_path, model_path