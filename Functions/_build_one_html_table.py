"""Build an HTML summary table (Table 1 layout: best window per model/horizon)."""

import pandas as pd


WINDOW_DISPLAY_NAMES = {
    'fixed120': 'Fixed 120',
    'fixed240': 'Fixed 240',
    'fixed300': 'Fixed 300',
    'expanding': 'Expanding',
}


def _build_one_html_table(
    df_ds: pd.DataFrame,
    table_model_order: list,
    table_expected_horizons: list,
    table_best_by: str,
    table_numeric_fmt: str,
    display_model_fn,
) -> str:
    """Return an HTML <table> string with rowspan on the Model column.

    Shows the best-window result for each (model, horizon) combination.
    """
    models_present = [m for m in table_model_order if m in set(df_ds['model'])]
    for extra in sorted(set(df_ds['model']) - set(table_model_order)):
        models_present.append(extra)

    table_rows = []
    for model_key in models_present:
        for horizon in table_expected_horizons:
            sub       = df_ds[(df_ds['model'] == model_key) & (df_ds['horizon_min'] == horizon)]
            sub_valid = sub.dropna(subset=[table_best_by]) if not sub.empty else sub
            if sub.empty or sub_valid.empty:
                table_rows.append({
                    'model_display': display_model_fn(model_key),
                    'Horizon': horizon,
                    'Best Window': '', 'RMSE': '', 'MAE': '', 'R2': '', 'Dir_acc': '',
                })
                continue
            if table_best_by == 'rmse':
                best = sub_valid.loc[sub_valid['rmse'].idxmin()]
            elif table_best_by == 'r2':
                best = sub_valid.loc[sub_valid['r2'].idxmax()]
            else:
                raise ValueError(f'Unknown table_best_by={table_best_by!r}')
            table_rows.append({
                'model_display': display_model_fn(model_key),
                'Horizon':  horizon,
                'Best Window': WINDOW_DISPLAY_NAMES.get(best['window'], best['window']),
                'RMSE':    table_numeric_fmt.format(best['rmse']),
                'MAE':     table_numeric_fmt.format(best['mae']),
                'R2':      table_numeric_fmt.format(best['r2']),
                'Dir_acc': table_numeric_fmt.format(best['dir_acc']),
            })

    css_table  = ('border-collapse: collapse; border: 1px solid black; '
                  'font-family: Arial, sans-serif; font-size: 12px;')
    css_header = ('border: 1px solid black; padding: 6px 12px; '
                  'font-weight: bold; text-align: left; background-color: #ffffff;')
    css_cell   = 'border: 1px solid black; padding: 6px 12px; text-align: left;'

    out = [f'<table style="{css_table}">']
    out.append('  <thead><tr>')
    for col in ['Model', 'Horizon', 'Best Window', 'RMSE', 'MAE', 'R2', 'Dir_acc']:
        out.append(f'    <th style="{css_header}">{col}</th>')
    out.append('  </tr></thead>')
    out.append('  <tbody>')

    i = 0
    while i < len(table_rows):
        model_label = table_rows[i]['model_display']
        j = i
        while j < len(table_rows) and table_rows[j]['model_display'] == model_label:
            j += 1
        group = table_rows[i:j]
        for k, row in enumerate(group):
            out.append('    <tr>')
            if k == 0:
                out.append(
                    f'      <td style="{css_cell}" rowspan="{len(group)}">{model_label}</td>'
                )
            out.append(f'      <td style="{css_cell}">{row["Horizon"]}</td>')
            out.append(f'      <td style="{css_cell}">{row["Best Window"]}</td>')
            out.append(f'      <td style="{css_cell}">{row["RMSE"]}</td>')
            out.append(f'      <td style="{css_cell}">{row["MAE"]}</td>')
            out.append(f'      <td style="{css_cell}">{row["R2"]}</td>')
            out.append(f'      <td style="{css_cell}">{row["Dir_acc"]}</td>')
            out.append('    </tr>')
        i = j

    out.append('  </tbody>')
    out.append('</table>')
    return '\n'.join(out)
