import re

import numpy as np
import pandas as pd


def make_stationary_features(df: pd.DataFrame, exclude_cols=None) -> pd.DataFrame:
    exclude_cols = set(exclude_cols or [])
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        series = pd.to_numeric(df[col], errors='coerce')
        if col in exclude_cols:
            out[col] = series
            continue
        if (series.dropna() > 0).all():
            out[f'{col}_logret'] = np.log(series).diff()
        else:
            out[f'{col}_diff'] = series.diff()
    return out


def build_X_traditional(
    bb_dict: dict,
    target_sheet: str,
    master_index: pd.DatetimeIndex,
    align_index: pd.DatetimeIndex,
    *,
    max_ffill_bars=None,
    use_staleness: bool = True,
    ar_lags=(1, 2, 3, 6, 12),
    ar_ma_windows=(3, 6, 12, 36),
    ar_vol_windows=(12, 36),
    logret_bar: pd.Series | None = None,
    stationarity_mode: str,
    stationary_panel_path=None,
    stationary_metadata=None,
) -> pd.DataFrame:
    if stationarity_mode == 'test_driven':
        if stationary_panel_path is None:
            raise FileNotFoundError(
                "TRADITIONAL_STATIONARITY_MODE='test_driven' but no Bloomberg stationary panel was loaded."
            )

        stationary = (
            pd.read_csv(stationary_panel_path, parse_dates=['Date'])
            .set_index('Date')
            .sort_index()
        )
        stationary = stationary.reindex(master_index)

        if stationary_metadata is not None:
            metadata_target_sheet = stationary_metadata.get('target_sheet')
            if metadata_target_sheet not in (None, target_sheet):
                print(
                    f"Warning: Bloomberg stationarity metadata target_sheet={metadata_target_sheet!r} "
                    f"does not match target_sheet={target_sheet!r}."
                )

        if not use_staleness:
            stationary = stationary.loc[:, [col for col in stationary.columns if not col.endswith('_stale')]]

    elif stationarity_mode == 'blanket':
        price_cols_raw = {}
        stale_cols_raw = {}

        for sheet, df in bb_dict.items():
            if sheet == target_sheet or 'close' not in df.columns:
                continue

            feature_name = re.sub(r'[^0-9a-zA-Z]+', '_', sheet).strip('_').lower()
            series = df['close'].sort_index()
            series = series[~series.index.duplicated(keep='last')]
            series = series.reindex(master_index)

            if use_staleness:
                is_new_tick = series.notna()
                real_tick_groups = is_new_tick.cumsum()
                stale_count = (~is_new_tick).groupby(real_tick_groups).cumsum().astype(int)
                stale_cols_raw[f'{feature_name}_stale'] = stale_count

            price_cols_raw[feature_name] = series.ffill(limit=max_ffill_bars)

        if not price_cols_raw:
            return pd.DataFrame(index=align_index)

        price_df = pd.DataFrame(price_cols_raw, index=master_index)
        stale_df = pd.DataFrame(stale_cols_raw, index=master_index)
        stationary = make_stationary_features(price_df)
        if use_staleness and not stale_df.empty:
            stationary = stationary.join(stale_df, how='left')

    else:
        raise ValueError(f'Unknown TRADITIONAL_STATIONARITY_MODE: {stationarity_mode!r}')

    if logret_bar is not None:
        logret_aligned = logret_bar.reindex(master_index)
        for lag in ar_lags:
            stationary[f'trad_gold_lag{lag}'] = logret_aligned.shift(lag)
        for window in ar_ma_windows:
            stationary[f'trad_gold_ma{window}'] = logret_aligned.shift(1).rolling(window).mean()
        for window in ar_vol_windows:
            stationary[f'trad_gold_std{window}'] = logret_aligned.shift(1).rolling(window).std()

    stationary = stationary.reindex(align_index)
    stationary = stationary.replace([np.inf, -np.inf], np.nan)
    stationary = stationary.ffill().fillna(0.0)
    return stationary
