import re
from hashlib import md5


def _slugify(text: str, maxlen: int = 55) -> str:
    text = text.lower()[:maxlen]
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _flat_col_name(var: str, market: str) -> str:
    """Stable slug: var__slugified-market_hash6. Mirrors the tabular export convention."""
    suffix = md5(market.encode("utf-8")).hexdigest()[:6]
    return f"{var}__{_slugify(market)}_{suffix}"