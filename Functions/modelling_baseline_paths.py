import re
from pathlib import Path


def resolve_bloomberg_xlsx(candidates: list[Path], data_dir: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    wildcard = sorted(data_dir.glob('*Indicators*Data*bloomberg*.xlsx'))
    if wildcard:
        return wildcard[0]
    tried = '\n  - '.join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        'Bloomberg workbook not found. Tried:\n'
        f'  - {tried}\n'
        "Expected something like 'Indicators Data bloomberg.xlsx' in ./Data."
    )


def resolve_latest_bloomberg_stationarity_artefacts(
    data_dir: Path,
    panel_glob: str,
    metadata_glob: str,
) -> tuple[Path, Path, str]:
    panel_rx = re.compile(r'bloomberg_panel_stationary_(\d{4}-\d{2}-\d{2})\.csv$')
    metadata_rx = re.compile(r'bloomberg_stationarity_metadata_(\d{4}-\d{2}-\d{2})\.json$')

    panel_candidates = []
    for path in data_dir.glob(panel_glob):
        match = panel_rx.search(path.name)
        if match:
            panel_candidates.append((match.group(1), path))

    if not panel_candidates:
        raise FileNotFoundError(
            f'No files matching {panel_glob} in {data_dir}.\n'
            'Run the PLS Feature Engineering notebook first to generate the Bloomberg stationary panel.'
        )

    metadata_by_date = {}
    for path in data_dir.glob(metadata_glob):
        match = metadata_rx.search(path.name)
        if match:
            metadata_by_date[match.group(1)] = path

    panel_candidates.sort(key=lambda item: item[0])
    latest_date, panel_path = panel_candidates[-1]
    metadata_path = metadata_by_date.get(latest_date)
    if metadata_path is None:
        raise FileNotFoundError(
            f'Missing matching bloomberg_stationarity_metadata_{{date}}.json for {panel_path.name}.'
        )

    return panel_path, metadata_path, latest_date


def find_latest_filtered_panel(folder: Path) -> tuple[Path, str]:
    pattern = 'polymarket_panel_filtered_*.csv'
    file_rx = re.compile(r'polymarket_panel_filtered_(\d{4}-\d{2}-\d{2})\.csv$')
    candidates = []
    for path in folder.glob(pattern):
        match = file_rx.search(path.name)
        if match:
            candidates.append((match.group(1), path))
    if not candidates:
        raise FileNotFoundError(
            f'No polymarket_panel_filtered_*.csv found in {folder}.\n'
            "Run the PLS Feature Engineering notebook first, or set FE_INPUT_MODE='preprocessed' only "
            'to use the dormant legacy pre-engineered-panel branch.'
        )
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1], candidates[-1][0]


def find_latest_panel(folder: Path, pattern: str) -> tuple[Path, str]:
    file_rx = re.compile(r'gold_panel_(\d{4}-\d{2}-\d{2})\.csv$')
    candidates = []
    for path in folder.glob(pattern):
        match = file_rx.search(path.name)
        if match:
            candidates.append((match.group(1), path))
    if not candidates:
        raise FileNotFoundError(f'No files matching {pattern} in {folder}')
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1], candidates[-1][0]
