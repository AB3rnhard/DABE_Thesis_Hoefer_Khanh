import pandas as pd


def _ensure_market_feature_multiindex(frame, frame_name, multiindex_names):
    if isinstance(frame.columns, pd.MultiIndex):
        return frame.sort_index(axis=1)

    if len(frame.columns) == 0:
        fixed = frame.copy()
        fixed.columns = pd.MultiIndex.from_tuples([], names=multiindex_names)
        return fixed

    if all(isinstance(column, tuple) and len(column) == 2 for column in frame.columns):
        fixed = frame.copy()
        fixed.columns = pd.MultiIndex.from_tuples(list(frame.columns), names=multiindex_names)
        print(f"Normalized {frame_name} columns back to a 2-level MultiIndex.")
        return fixed.sort_index(axis=1)

    raise TypeError(
        f"{frame_name} must have 2-level (variable, market) columns; got {type(frame.columns).__name__}."
    )