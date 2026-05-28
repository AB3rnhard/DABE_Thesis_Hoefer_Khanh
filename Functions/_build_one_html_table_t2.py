"""Build an HTML summary table (Table 2 layout: every window per model/horizon)."""

import pandas as pd


WINDOW_DISPLAY_NAMES_T2 = {
    'fixed120':  'Fixed 120',
    'fixed240':  'Fixed 240',
    'fixed300':  'Fixed 300',
    'expanding': 'Expanding',
}


def _build_one_html_table_t2(
    df_ds: pd.DataFrame,
    table2_model_order: list,
    table2_expected_horizons: list,
    table2_expected_windows: list,
    table2_numeric_fmt: str,
) -> str:
    """Return HTML with rowspan on both Model and Horizon columns.

    Expands every window scheme sepperately rather than collapsing to the best.
    """
    models_present = [m for m in table2_model_order if m in set(df_ds['model'])]
    for extra in sorted(set(df_ds['model']) - set(table2_model_order)):
        models_present.append(extra)

    _row_lookup = {
        (str(r['model']), int(r['horizon_min']), str(r['window'])): r
        for _, r in df_ds.iterrows()
    }

    n_windows  = len(table2_expected_windows)
    table_rows = []
    for model_key in models_present:
        for h in table2_expected_horizons:
            for window_key in table2_expected_windows:
                hit = _row_lookup.get((model_key, int(h), window_key))
                row = {
                    'model_display': model_key,
                    'Horizon': h,
                    'Window':  WINDOW_DISPLAY_NAMES_T2.get(window_key, window_key),
                    'RMSE': '', 'MAE': '', 'R2': '', 'Dir_acc': '',
                }
                if hit is not None and pd.notna(hit.get('rmse')):
                    row['RMSE']    = table2_numeric_fmt.format(hit['rmse'])
                    row['MAE']     = table2_numeric_fmt.format(hit['mae'])
                    row['R2']      = table2_numeric_fmt.format(hit['r2'])
                    row['Dir_acc'] = table2_numeric_fmt.format(hit['dir_acc'])
                table_rows.append(row)

    css_table  = ('border-collapse: collapse; border: 1px solid black; '
                  'font-family: Arial, sans-serif; font-size: 12px;')
    css_header = ('border: 1px solid black; padding: 6px 12px; '
                  'font-weight: bold; text-align: left; background-color: #ffffff;')
    css_cell   = 'border: 1px solid black; padding: 6px 12px; text-align: left;'

    out = [f'<table style="{css_table}">']
    out.append('  <thead><tr>')
    for col in ['Model', 'Horizon', 'Window', 'RMSE', 'MAE', 'R2', 'Dir_acc']:
        out.append(f'    <th style="{css_header}">{col}</th>')
    out.append('  </tr></thead>')
    out.append('  <tbody>')

    i = 0
    while i < len(table_rows):
        model_label = table_rows[i]['model_display']
        j = i
        while j < len(table_rows) and table_rows[j]['model_display'] == model_label:
            j += 1
        model_group   = table_rows[i:j]
        model_rowspan = len(model_group)

        k = 0
        while k < len(model_group):
            horizon_label = model_group[k]['Horizon']
            l = k
            while l < len(model_group) and model_group[l]['Horizon'] == horizon_label:
                l += 1
            horizon_group   = model_group[k:l]
            horizon_rowspan = len(horizon_group)

            for m_idx, row in enumerate(horizon_group):
                out.append('    <tr>')
                if k == 0 and m_idx == 0:
                    out.append(
                        f'      <td style="{css_cell}" rowspan="{model_rowspan}">{model_label}</td>'
                    )
                if m_idx == 0:
                    out.append(
                        f'      <td style="{css_cell}" rowspan="{horizon_rowspan}">{horizon_label}</td>'
                    )
                out.append(f'      <td style="{css_cell}">{row["Window"]}</td>')
                out.append(f'      <td style="{css_cell}">{row["RMSE"]}</td>')
                out.append(f'      <td style="{css_cell}">{row["MAE"]}</td>')
                out.append(f'      <td style="{css_cell}">{row["R2"]}</td>')
                out.append(f'      <td style="{css_cell}">{row["Dir_acc"]}</td>')
                out.append('    </tr>')
            k = l
        i = j

    out.append('  </tbody>')
    out.append('</table>')
    return '\n'.join(out)
