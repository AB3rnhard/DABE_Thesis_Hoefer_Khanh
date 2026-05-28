import re

_MONTH_ABBR = {
    'january': 'Jan', 'february': 'Feb', 'march': 'Mar', 'april': 'Apr',
    'may': 'May', 'june': 'Jun', 'july': 'Jul', 'august': 'Aug',
    'september': 'Sep', 'october': 'Oct', 'november': 'Nov', 'december': 'Dec',
}
_SKIP_WORDS = {'will', 'gc', 'of', 'the'}
_WORD_MAP = {
    'gold': 'Gold', 'hit': 'hit', 'high': '>',   'low': '<',
    'over': '>',    'under': '<', 'settle': 'settle',
    'at': 'at',     'by': 'by',   'end': 'end',  'in': 'in', 'on': 'on',
}


def _parse_flat_column(col: str):
    """Split '<var>__<slug>_<hash6>' -> (var, market_id)."""
    parts = col.split('__')
    if len(parts) < 2:
        return col, ''
    return '__'.join(parts[:-1]), parts[-1]


def _prettify_market(slug: str) -> str:
    """Convert a Polymarket slug into a concise human-readable label.

    Strips the trailing hash, merges digit tokens into formatted numbers,
    and maps common words to shorter equivalents.
    """
    slug = re.sub(r'_[0-9a-f]{6}$', '', slug)
    tokens = slug.split('_')

    # Merge digit tokens into comma-formatted numbers.
    merged = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.isdigit():
            parts = [t]
            while (i + 1 < len(tokens)
                   and tokens[i + 1].isdigit()
                   and len(tokens[i + 1]) == 3):
                i += 1
                parts.append(tokens[i])
            merged.append(f'{int("".join(parts)):,}')
        else:
            merged.append(t)
        i += 1

    # Join two consecutive formatted numbers with an en-dash -> price range.
    joined = []
    i = 0
    _is_num = lambda s: bool(re.fullmatch(r'[\d,]+', s))
    while i < len(merged):
        if _is_num(merged[i]) and i + 1 < len(merged) and _is_num(merged[i + 1]):
            joined.append(merged[i] + '\u2013' + merged[i + 1])
            i += 2
        else:
            joined.append(merged[i])
            i += 1

    result = []
    for t in joined:
        tl = t.lower()
        if tl in _SKIP_WORDS:
            continue
        elif tl in _MONTH_ABBR:
            result.append(_MONTH_ABBR[tl])
        elif tl in _WORD_MAP:
            result.append(_WORD_MAP[tl])
        else:
            result.append(t)

    return ' '.join(result).strip()
