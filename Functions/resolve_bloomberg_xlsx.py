from pathlib import Path


def resolve_bloomberg_xlsx(candidates: list[Path], data_dir: Path) -> Path:
    for path in candidates:
        if path.exists():
            return path
    wildcard = sorted(data_dir.glob("*Indicators*Data*bloomberg*.xlsx"))
    if wildcard:
        return wildcard[0]
    tried = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Bloomberg workbook not found. Tried:\n"
        f"  - {tried}\n"
        "Expected something like 'Indicators Data bloomberg.xlsx' in ./Data."
    )