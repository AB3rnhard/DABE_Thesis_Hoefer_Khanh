import pandas as pd


def q(sql, conn, label=None):
    """Run a SQL query and return a formatted DataFrame."""
    if label:
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}")
    df = pd.read_sql_query(sql, conn)
    print(df.to_string(index=False))
    return df