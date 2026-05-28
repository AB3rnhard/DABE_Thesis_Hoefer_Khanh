"""Utility functions for loading, normalising, and ordering modelling results."""

import json
import re
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Canonical orderings (shared with the notebook via import)
# ---------------------------------------------------------------------------
CANONICAL_MODEL_ORDER   = ['OLS', 'RF', 'LSTM']
CANONICAL_DATASET_ORDER = [
    'poly', 'bloomberg', 'ar',
    'poly_ar', 'bloom_ar', 'poly_bloom_ar',
]
CANONICAL_HORIZONS = [10, 15, 30, 45, 60]

MODEL_NAME_MAP = {
    'linear': 'OLS', 'pls_ols': 'OLS',
    'rf': 'RF',     'rf_pls': 'RF',
    'lstm': 'LSTM', 'lstm_pls': 'LSTM',
}
DATASET_NAME_MAP = {
    'ar_only': 'ar', 'bloomberg_only': 'bloomberg',
    'poly': 'poly_ar', 'poly+trad': 'poly_bloom_ar',
    'poly_only': 'poly', 'trad': 'bloom_ar', 'bloomberg_ar': 'bloom_ar',
}
MODEL_DISPLAY_NAMES = {name: name for name in CANONICAL_MODEL_ORDER}
DATASET_DISPLAY_NAMES = {
    'poly': 'poly', 'bloomberg': 'bloomberg', 'ar': 'ar',
    'poly_ar': 'poly_ar', 'bloom_ar': 'bloomberg_ar',
    'poly_bloom_ar': 'poly_bloom_ar',
}

_SUMMARY_RX = re.compile(r'^summary_h(\d+)m_(\d{4}-\d{2}-\d{2})\.csv$')
_PRED_RX = re.compile(
    r'^(?P<model>.+?)_(?P<window>fixed\d+|expanding)_(?P<dataset>[a-z0-9+_]+)'
    r'_h(?P<h>\d+)m_f(?P<f>\d+)_(?P<date>\d{4}-\d{2}-\d{2})$'
)


def load_summary_files(results_dir: Path) -> dict:
    """Return {horizon_min: DataFrame} for every summary CSV found in results_dir."""
    out: dict = {}
    for path in sorted(results_dir.glob('summary_h*m_*.csv')):
        m = _SUMMARY_RX.match(path.name)
        if not m:
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f'  [warn] {path.name}: {exc}')
            continue
        if df.empty:
            continue
        h = int(m.group(1))
        if h in out and out[h].attrs.get('date', '') > m.group(2):
            continue
        df.attrs['date'] = m.group(2)
        df.attrs['path'] = str(path)
        out[h] = df
    return out


def discover_prediction_files(results_dir: Path) -> dict:
    """Map (model, window, dataset, horizon_min) -> latest predictions.csv path."""
    out: dict = {}
    best_date: dict = {}
    for path in sorted(results_dir.glob('*.predictions.csv')):
        stem = path.name[:-len('.predictions.csv')]
        m = _PRED_RX.match(stem)
        if not m:
            continue
        key = (m['model'].lower(), m['window'], m['dataset'], int(m['h']))
        if best_date.get(key, '') < m['date']:
            best_date[key] = m['date']
            out[key] = path
    return out


def discover_model_files(models_dir: Path) -> dict:
    """Map (model, window, dataset, horizon_min) -> latest .pkl path."""
    out: dict = {}
    best_date: dict = {}
    if not models_dir.exists():
        return out
    for path in sorted(models_dir.glob('*.pkl')):
        m = _PRED_RX.match(path.stem)
        if not m:
            continue
        key = (m['model'].lower(), m['window'], m['dataset'], int(m['h']))
        if best_date.get(key, '') < m['date']:
            best_date[key] = m['date']
            out[key] = path
    return out


def build_summary_from_run_metadata(results_dir: Path) -> pd.DataFrame:
    """Parse every per-run JSON sidecar in results_dir.

    Each sidecar is the source-of-truth for its (model, window, dataset,
    horizon) combination. On duplicate keys, the most recent data_date wins.
    """
    by_key: dict = {}
    by_key_date: dict = {}
    for path in sorted(results_dir.glob('*.json')):
        if path.name.endswith('.predictions.json'):
            continue
        m = _PRED_RX.match(path.stem)
        if not m:
            continue
        try:
            meta = json.loads(path.read_text(encoding='utf-8'))
        except Exception as exc:
            print(f'  [warn] could not parse {path.name}: {exc}')
            continue
        metrics = meta.get('metrics') or {}
        config  = meta.get('config') or {}
        horizon_min = config.get('return_horizon_min') or int(m['h'])
        key = (
            meta.get('dataset_tag', m['dataset']),
            str(meta.get('model', m['model'].lower())).lower(),
            meta.get('window', m['window']),
            int(horizon_min),
        )
        cmp_key = (str(meta.get('data_date', m['date'])), str(meta.get('run_utc', '')))
        if key in by_key and by_key_date[key] >= cmp_key:
            continue
        by_key_date[key] = cmp_key
        by_key[key] = {
            'dataset':     key[0],
            'model':       key[1],
            'window':      key[2],
            'date':        meta.get('data_date', m['date']),
            'horizon_min': int(horizon_min),
            'n_features':  meta.get('n_features', int(m['f'])),
            'rmse':        metrics.get('rmse'),
            'mae':         metrics.get('mae'),
            'r2':          metrics.get('r2'),
            'dir_acc':     metrics.get('dir_acc'),
            'n':           metrics.get('n'),
            'train_time_s': meta.get('train_time_s'),
        }
    return pd.DataFrame(list(by_key.values()))


# ---------------------------------------------------------------------------
# Ordering + display helpers
# ---------------------------------------------------------------------------

def _ordered_categories(values, canonical_order):
    present = [v for v in canonical_order if v in values]
    extras  = sorted(v for v in values if v not in canonical_order)
    return present + extras


def _ordered_horizons(values):
    values = sorted({int(v) for v in values if pd.notna(v)})
    return [h for h in CANONICAL_HORIZONS if h in values] + [h for h in values if h not in CANONICAL_HORIZONS]


def _dataset_order(values):
    return _ordered_categories(
        {str(v) for v in values if pd.notna(v)}, CANONICAL_DATASET_ORDER
    )


def _model_order(values):
    return _ordered_categories(
        {str(v) for v in values if pd.notna(v)}, CANONICAL_MODEL_ORDER
    )


def _display_dataset(dataset):
    return DATASET_DISPLAY_NAMES.get(str(dataset), str(dataset))


def _display_model(model):
    return MODEL_DISPLAY_NAMES.get(str(model), str(model))


def _sort_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Sort df by horizon, dataset, model in canonical order."""
    if df.empty:
        return df.copy()
    out = df.copy()
    tmp = []
    if 'horizon_min' in out.columns:
        h_ord = {h: i for i, h in enumerate(_ordered_horizons(out['horizon_min']))}
        out['_h_order'] = out['horizon_min'].astype(int).map(h_ord)
        tmp.append('_h_order')
    if 'dataset' in out.columns:
        d_ord = {d: i for i, d in enumerate(_dataset_order(out['dataset']))}
        out['_d_order'] = out['dataset'].astype(str).map(d_ord)
        tmp.append('_d_order')
    if 'model' in out.columns:
        m_ord = {m: i for i, m in enumerate(_model_order(out['model']))}
        out['_m_order'] = out['model'].astype(str).map(m_ord)
        tmp.append('_m_order')
    sort_cols = [c for c in ['_h_order', '_d_order', '_m_order', 'window'] if c in out.columns]
    out = out.sort_values(sort_cols, kind='stable')
    return out.drop(columns=tmp, errors='ignore').reset_index(drop=True)


def _normalize_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['model']   = df['model'].astype(str).str.strip().str.lower().map(
        lambda m: MODEL_NAME_MAP.get(m, m.upper())
    )
    df['dataset'] = df['dataset'].astype(str).str.strip().map(
        lambda d: DATASET_NAME_MAP.get(d, d)
    )
    return df


def _renormalize_file_dict(d: dict) -> dict:
    """Re-key a (model, window, dataset, horizon) file dict after name normalisation."""
    out: dict = {}
    for (model, window, dataset, horizon), path in d.items():
        nm = MODEL_NAME_MAP.get(str(model).strip().lower(), str(model).strip().upper())
        nd = DATASET_NAME_MAP.get(str(dataset).strip(), str(dataset).strip())
        key = (nm, window, nd, horizon)
        if key not in out or str(out[key]) < str(path):
            out[key] = path
    return out
