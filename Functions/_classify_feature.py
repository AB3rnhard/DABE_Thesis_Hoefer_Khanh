"""Classify a feature name into a broad family group for colouring."""


def _classify_feature(name: str) -> str:
    """Return a coarse family label for *name* (used only for plot colours)."""
    name_l = str(name).lower()
    if name_l.startswith('ar_') or name_l.startswith('trad_gold_'):
        return 'AR (gold returns)'
    if name_l in {'is_post_break', 'poly_movement_during_break',
                  'trad_movement_during_break'}:
        return 'Gap features'
    if (
        name_l.endswith('_logret') or name_l.endswith('_stale')
        or name_l.endswith('_diff')
        or any(key in name_l for key in ('crude_oil', 's_p_500', 'usd_index',
                                          'vix_index', 'usgg10yr', 'eurusd',
                                          'gc1_comdty', 'gld_us_equity'))
        or name_l.startswith('trad__')
    ):
        return 'Bloomberg (macro)'
    return 'Polymarket'
