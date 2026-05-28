import re
from pathlib import Path

_DATE_RX = re.compile(r'(\d{4}-\d{2}-\d{2})')


def find_latest_file(folder: Path, pattern: str):
    """Return (path, date_str) for the newest date-stamped file matching pattern."""
    candidates = []
    for p in folder.glob(pattern):
        m = _DATE_RX.search(p.name)
        if m:
            candidates.append((m.group(1), p))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1], candidates[-1][0]
