import pandas as pd
from pathlib import Path


def load_bloomberg(xlsx_path: Path) -> dict:
    """Load the Bloomberg workbook; normalise the close column to 'close'.

    Mirrors the equivelant function in Modelling Baseline.
    """
    xl = pd.ExcelFile(xlsx_path)
    out = {}
    for sheet in xl.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet, header=0)
        if df.empty or 'Date' not in df.columns:
            continue
        df = df.rename(columns={c: c.strip() for c in df.columns})
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']).set_index('Date').sort_index()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'Close' in df.columns:
            df = df.rename(columns={'Close': 'close'})
        elif 'Last Price' in df.columns:
            df = df.rename(columns={'Last Price': 'close'})
        else:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not numeric_cols:
                continue
            df = df.rename(columns={numeric_cols[-1]: 'close'})
        out[sheet] = df
    return out
