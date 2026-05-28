import ast
import json
import re

import pandas as pd


def _parse_numeric_list(value):
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    parsed = None
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        pass
    if parsed is None:
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
    if isinstance(parsed, (list, tuple)):
        result = []
        for item in parsed:
            try:
                result.append(float(item))
            except (TypeError, ValueError):
                pass
        return result if result else None
    if isinstance(parsed, (int, float)) and not isinstance(parsed, bool):
        return [float(parsed)]
    return None


def _safe_feature_name(text):
    text = str(text).strip().lower()
    text = re.sub(r'[^0-9a-zA-Z]+', '_', text).strip('_')
    return text or 'value'


def _parse_label_list(value):
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return None


def _expand_json_numeric_columns(sub, var_name, original_predictor_frame, multiindex_names):
    expanded_cols = {}
    label_sub = None
    if var_name == 'outcomePrices' and 'outcomes' in original_predictor_frame.columns.get_level_values(0):
        try:
            label_sub = original_predictor_frame['outcomes'].reindex(columns=sub.columns)
        except Exception:
            label_sub = None

    for market in sub.columns:
        series = sub[market]
        parsed_rows = {}
        label_series = None
        if label_sub is not None:
            if isinstance(label_sub, pd.Series):
                if market == label_sub.name:
                    label_series = label_sub
            elif market in label_sub.columns:
                label_series = label_sub[market]

        for idx, raw_value in series.dropna().items():
            try:
                parsed_values = _parse_numeric_list(raw_value)
            except Exception:
                continue
            if parsed_values is None:
                continue
            if not isinstance(parsed_values, (list, tuple)):
                continue
            if len(parsed_values) == 0:
                continue
            labels = None
            if label_series is not None and idx in label_series.index:
                labels = _parse_label_list(label_series.loc[idx])
            for pos, numeric_value in enumerate(parsed_values):
                numeric_value = pd.to_numeric(numeric_value, errors='coerce')
                if pd.isna(numeric_value):
                    continue
                if labels is not None and pos < len(labels):
                    suffix = _safe_feature_name(labels[pos])
                else:
                    suffix = f'json_{pos}'
                derived_var = f'{var_name}__{suffix}'
                parsed_rows.setdefault(derived_var, {})[idx] = float(numeric_value)
        for derived_var, row_map in parsed_rows.items():
            expanded_cols[(derived_var, market)] = pd.Series(
                row_map, index=sub.index, dtype='float64'
            )
    if not expanded_cols:
        return None
    expanded_df = pd.DataFrame(expanded_cols, index=sub.index)
    expanded_df.columns = pd.MultiIndex.from_tuples(
        expanded_df.columns, names=multiindex_names
    )
    return expanded_df.sort_index(axis=1)