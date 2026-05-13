# %% [markdown]
# ### Implementation with 300 seconds timer

# %%
import json
import sqlite3
import time
import logging
import re
import signal
import os
from datetime import datetime
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ── PATHS & CONSTANTS ──────────────────────────────────────────────────────────
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
DB_PATH = BASE_DIR / "Data" / "polymarket_gamma_dynamic.sqlite"
LOG_PATH = BASE_DIR / "logs" / "scraper.log"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

NUMERIC_COLUMNS = {
    "volume", "volumeNum", "liquidity", "liquidityNum",
    "lastTradePrice", "bestBid", "bestAsk",
    "spread", "volume24hr", "volume1wk", "volume1mo",
}

# Common "missing" sentinel strings — map to None without raising a coercion failure.
# These are NOT counted as coerce_failures; they are expected absence signals from the API.
MISSING_NUMERIC_TOKENS = {"", "none", "null", "nan", "n/a", "na", "-", "--", "unknown"}
NUMERIC_SUFFIX_MULTIPLIERS = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}

KEY_FIELDS_TO_MONITOR     = {"volume", "lastTradePrice", "bestBid", "bestAsk"}
NULL_RATE_ALERT_THRESHOLD = 0.10   # warn if >10% of rows are null on a key field


# ── LOGGING ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ── LIFECYCLE STATE ────────────────────────────────────────────────────────────
_consecutive_failures             = 0
_run_count                        = 0
CONSECUTIVE_FAILURE_ALERT_THRESHOLD = 3
# FIX: Hard stop after this many consecutive failures so an external supervisor
# (systemd, cron, container) can detect the stall and restart/alert.
# Set to None to disable hard stop and keep the original "spin forever" behaviour.
MAX_CONSECUTIVE_FAILURES          = 10
HEARTBEAT_EVERY_N_RUNS            = 12   # every ~1 hour at 5-min intervals
# FIX: Warn if a single run takes longer than this fraction of the interval,
# so you know you are approaching overlap between scheduled runs.
RUN_TIME_WARN_SECONDS             = 240  # warn when a run exceeds 240s of the 300s window


# ── HTTP SESSION ───────────────────────────────────────────────────────────────
def get_retrying_session():
    session = requests.Session()
    session.headers.update({"User-Agent": "PolymarketGoldScraper/1.0"})
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://",  adapter)
    session.mount("https://", adapter)
    return session


http_session = get_retrying_session()


# ── TAG CACHE ──────────────────────────────────────────────────────────────────
CACHED_TAG_IDS       = []
TAG_CACHE_LAST_FETCH = 0.0
TAG_CACHE_TTL        = 3600   # refresh every hour


# ── DATABASE ───────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


# ── PYTHONANYWHERE CPU MONITORING ─────────────────────────────────────────────
PYTHONANYWHERE_USERNAME = os.getenv("PYTHONANYWHERE_USERNAME")
PYTHONANYWHERE_API_TOKEN = os.getenv("PYTHONANYWHERE_API_TOKEN")
PYTHONANYWHERE_CPU_TIMEOUT = (5, 15)


def log_pythonanywhere_cpu_usage():
    """Fetches and logs PythonAnywhere daily CPU usage. Never raises."""
    if not PYTHONANYWHERE_USERNAME or not PYTHONANYWHERE_API_TOKEN:
        logger.info("PythonAnywhere CPU usage check skipped (missing env vars).")
        return

    cpu_url = f"https://www.pythonanywhere.com/api/v0/user/{PYTHONANYWHERE_USERNAME}/cpu/"
    headers = {"Authorization": f"Token {PYTHONANYWHERE_API_TOKEN}"}

    try:
        response = http_session.get(cpu_url, headers=headers, timeout=PYTHONANYWHERE_CPU_TIMEOUT)
        response.raise_for_status()
        cpu_data = response.json()

        used = float(cpu_data.get("daily_cpu_total_usage_seconds", 0.0))
        limit = float(cpu_data.get("daily_cpu_limit", 0.0))
        usage_pct = (used / limit * 100.0) if limit > 0 else 0.0

        logger.info(
            f"PythonAnywhere CPU used: {used:.0f}s / {limit:.0f}s ({usage_pct:.1f}%)."
        )
    except Exception as e:
        logger.warning(f"Could not fetch PythonAnywhere CPU usage: {e}")


# ── NUMERIC COERCION ───────────────────────────────────────────────────────────
def _coerce_numeric_value(raw_value):
    """
    Relaxed numeric parsing designed to survive API drift.

    Returns (value, failed, used_relaxed):
      value        – float or None
      failed       – True only when a non-sentinel, non-parseable string was received
                     (these are counted as coerce_failures and logged as warnings)
      used_relaxed – True when the plain float() cast failed but the regex fallback
                     recovered a value (logged at INFO level for observability)

    NOTE on booleans: bool is a subclass of int in Python, so True -> 1.0 and
    False -> 0.0 when a boolean reaches a NUMERIC_COLUMNS field. This is intentional
    (e.g. binary outcome flags stored as 0/1), but if you see unexpected 0.0/1.0
    values, check whether the source field is actually boolean in the API response.
    """
    if raw_value is None:
        return None, False, False

    # bool check must come before the int/float check because bool is a subclass of int
    if isinstance(raw_value, bool):
        return float(raw_value), False, False

    if isinstance(raw_value, (int, float)):
        try:
            return float(raw_value), False, False
        except (ValueError, TypeError):
            return None, True, False

    text = str(raw_value).strip()
    if text.lower() in MISSING_NUMERIC_TOKENS:
        return None, False, False   # expected absence — NOT a coercion failure

    # Fast path: plain float cast
    try:
        return float(text), False, False
    except (ValueError, TypeError):
        pass

    # Relaxed path: strip formatting characters and apply regex
    normalized = text.replace(",", "").replace("_", "")
    if normalized.startswith("(") and normalized.endswith(")"):
        normalized = "-" + normalized[1:-1].strip()

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", normalized)
    if not match:
        return None, True, True   # genuinely unparseable — count as failure

    try:
        value = float(match.group(0))
    except ValueError:
        return None, True, True

    suffix_match = re.match(r"\s*([kKmMbBtT])", normalized[match.end():])
    if suffix_match:
        value *= NUMERIC_SUFFIX_MULTIPLIERS[suffix_match.group(1).lower()]

    if "%" in normalized:
        value /= 100.0

    return value, False, True   # recovered — not a failure, but used relaxed path


# ── SCRAPER ────────────────────────────────────────────────────────────────────
def gamma_scraper(endpoint="markets", fetch_all=True, **kwargs):
    """
    Returns [] on genuine empty response, None on hard failure.
    Callers can distinguish a broken request from an API returning no data.

    FIX (params.update bug): the original code called params.update(kwargs) AFTER
    building params from kwargs, which silently overwrote the carefully string-cast
    'active' and 'closed' values with raw Python booleans (True instead of "true"),
    and also redundantly re-set 'limit' and 'offset'. Now extra kwargs (e.g. tag_id)
    are merged separately, never clobbering the pre-processed keys.
    """
    base_url = f"https://gamma-api.polymarket.com/{endpoint}"

    # Pre-process the four known query parameters with explicit type handling
    _known_keys = {"limit", "offset", "active", "closed"}
    params = {
        "limit":  kwargs.get("limit",  100),
        "offset": kwargs.get("offset", 0),
        "active": str(kwargs.get("active", "true")).lower(),
        "closed": str(kwargs.get("closed", "false")).lower(),
    }
    # Merge any extra kwargs (e.g. tag_id) WITHOUT overwriting the processed keys above
    params.update({k: v for k, v in kwargs.items() if k not in _known_keys})

    try:
        if not fetch_all:
            response = http_session.get(base_url, params=params, timeout=(5, 30))
            response.raise_for_status()
            return response.json()

        all_items = []
        limit  = int(params["limit"])
        offset = int(params["offset"])
        start_offset = offset

        while True:
            current_params = dict(params)
            current_params["limit"]  = limit
            current_params["offset"] = offset

            response = http_session.get(base_url, params=current_params, timeout=(5, 30))
            response.raise_for_status()
            page_data = response.json()

            if isinstance(page_data, list):
                page_items = page_data
            elif isinstance(page_data, dict):
                # FIX: Accept wrapped payloads like {'markets': [...]} to survive API shape drift.
                page_items = None
                for key in (endpoint, "data", "results", "items", "rows"):
                    candidate = page_data.get(key)
                    if isinstance(candidate, list):
                        page_items = candidate
                        if offset == start_offset:
                            logger.info(
                                f"/{endpoint}: using wrapped list payload from key '{key}' "
                                f"(API response type=dict)."
                            )
                        break
                if page_items is None:
                    logger.warning(
                        f"/{endpoint}: expected list page, got dict without list payload. "
                        f"Returning as-is (possible API shape change)."
                    )
                    return page_data
            else:
                logger.warning(
                    f"/{endpoint}: expected list page, got {type(page_data).__name__}. "
                    f"Returning as-is (single-object response or API shape change)."
                )
                return page_data

            all_items.extend(page_items)
            if len(page_items) < limit:
                break
            offset += limit

        return all_items   # [] is a valid, non-error empty result

    except Exception as e:
        logger.error(f"Error fetching /{endpoint}: {e}")
        return None   # None signals a hard failure to the caller


# ── PIPELINE ───────────────────────────────────────────────────────────────────
def save_pipeline_dynamic(conn, data, table_name="markets"):
    """Saves a batch to SQLite with typed columns, write retry, and quality monitoring."""
    if not data:
        return 0

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    for item in data:
        item["scraped_at"] = current_time

    cursor   = conn.cursor()
    all_keys = set()
    for item in data:
        all_keys.update(item.keys())

    columns = sorted(list(all_keys))
    if "scraped_at" in columns: columns.remove("scraped_at")
    if "id"         in columns: columns.remove("id")
    columns.insert(0, "id")
    columns.insert(1, "scraped_at")

    def _col_type(col):
        return "REAL" if col in NUMERIC_COLUMNS else "TEXT"

    col_defs = [f'"{col}" {_col_type(col)}' for col in columns]
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {", ".join(col_defs)},
            PRIMARY KEY ("id", "scraped_at")
        )
    """)

    if table_name == "markets":
        cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_scraped_at ON {table_name}("scraped_at")')
        cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_id ON {table_name}("id")')

    cursor.execute(f"PRAGMA table_info({table_name})")
    existing_columns = {row[1] for row in cursor.fetchall()}
    for col in columns:
        if col not in existing_columns:
            cursor.execute(f'ALTER TABLE {table_name} ADD COLUMN "{col}" {_col_type(col)}')

    coerce_failures        = 0
    coerce_failures_by_col = {}
    relaxed_parse_count    = 0
    rows_to_insert         = []

    for item in data:
        row = []
        for col in columns:
            val = item.get(col)
            if col in NUMERIC_COLUMNS:
                coerced_val, failed, used_relaxed = _coerce_numeric_value(val)
                row.append(coerced_val)
                if failed:
                    coerce_failures += 1
                    coerce_failures_by_col[col] = coerce_failures_by_col.get(col, 0) + 1
                elif used_relaxed:
                    relaxed_parse_count += 1
            elif isinstance(val, (dict, list, bool)):
                # FIX: bool is handled here only for non-numeric columns.
                # For numeric columns, _coerce_numeric_value handles booleans above (True->1.0).
                row.append(json.dumps(val))
            else:
                row.append(val)
        rows_to_insert.append(tuple(row))

    if relaxed_parse_count > 0:
        logger.info(f"[{table_name}] Relaxed numeric parsing recovered {relaxed_parse_count} value(s).")

    if coerce_failures > 0:
        by_col = ", ".join(
            f"{col}={count}"
            for col, count in sorted(coerce_failures_by_col.items(), key=lambda x: (-x[1], x[0]))
        )
        logger.warning(
            f"[{table_name}] {coerce_failures} numeric coercion failure(s) — stored as NULL."
            + (f" Breakdown: {by_col}" if by_col else "")
        )

    # FIX: Null-rate check uses post-coercion rows_to_insert (not raw item dicts),
    # so the count accurately reflects what is actually stored in the DB.
    # Sentinel strings like "N/A" become None after coercion and are counted here.
    if table_name == "markets":
        n = len(rows_to_insert)
        for field in KEY_FIELDS_TO_MONITOR:
            if field in columns:
                field_idx  = columns.index(field)
                null_count = sum(1 for row in rows_to_insert if row[field_idx] is None)
                null_rate  = null_count / n
                if null_rate > NULL_RATE_ALERT_THRESHOLD:
                    logger.warning(
                        f"[{table_name}] High null rate on '{field}': "
                        f"{null_rate:.0%} ({null_count}/{n} rows) — "
                        f"likely genuine API absence; filter at modeling time."
                    )

    insert_sql = (
        f'INSERT OR IGNORE INTO {table_name} '
        f'({", ".join(f"{chr(34)}{c}{chr(34)}" for c in columns)}) '
        f'VALUES ({", ".join(["?"] * len(columns))})'
    )

    for attempt in range(3):
        try:
            cursor.executemany(insert_sql, rows_to_insert)
            conn.commit()
            break
        except sqlite3.OperationalError as e:
            if attempt < 2:
                wait = 2 ** attempt
                logger.warning(f"[{table_name}] DB write error (attempt {attempt+1}/3): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"[{table_name}] DB write failed after 3 attempts.", exc_info=True)
                raise

    rows_inserted = cursor.rowcount
    rows_ignored  = len(rows_to_insert) - rows_inserted
    if rows_ignored > 0:
        logger.info(f"[{table_name}] {rows_inserted} new rows inserted, {rows_ignored} duplicate rows ignored.")

    return rows_inserted


# ── JOB ────────────────────────────────────────────────────────────────────────
def run_job(keyword="gold", limit_results=None):
    """Single scraping run with lifecycle monitoring, quality checks, and failure alerting."""
    global CACHED_TAG_IDS, TAG_CACHE_LAST_FETCH, _consecutive_failures, _run_count

    _run_count += 1
    limit_text = "All" if limit_results is None else limit_results
    logger.info(f"[Run #{_run_count}] Starting scrape (keyword='{keyword}', limit={limit_text})...")

    # FIX: Track wall-clock time of the run to warn if it is eating into the next interval
    run_start = time.time()

    if _run_count % HEARTBEAT_EVERY_N_RUNS == 0:
        logger.info(
            f"[HEARTBEAT] Run #{_run_count} — scraper alive. "
            f"Consecutive failures: {_consecutive_failures}."
        )

    try:
        conn = init_db()

        cache_age = time.time() - TAG_CACHE_LAST_FETCH
        if not CACHED_TAG_IDS or cache_age > TAG_CACHE_TTL:
            logger.info(f"Tag cache stale ({int(cache_age)}s old). Fetching from API...")
            all_tags = gamma_scraper(endpoint="tags", limit=100)
            if all_tags is None:
                raise RuntimeError("Tag fetch returned None (request failed). Cannot proceed.")
            # FIX: Guard against unexpected non-list tag response before iterating
            if not isinstance(all_tags, list):
                raise RuntimeError(
                    f"Tag fetch returned unexpected type {type(all_tags).__name__} "
                    f"(expected list). Cannot build tag cache."
                )
            target_keywords = ["finance", "crypto", "geopolitics", "politics"]
            CACHED_TAG_IDS = list({
                tag.get("id")
                for tag in all_tags
                if isinstance(tag, dict) and any(
                    tk in str(tag.get("label", "")).lower() or
                    tk in str(tag.get("slug",  "")).lower()
                    for tk in target_keywords
                )
            })
            TAG_CACHE_LAST_FETCH = time.time()
            save_pipeline_dynamic(conn, all_tags, "tags")
            logger.info(f"Cached {len(CACHED_TAG_IDS)} tag IDs for the next {TAG_CACHE_TTL}s.")
        else:
            logger.info(f"Using {len(CACHED_TAG_IDS)} cached tag IDs ({int(cache_age)}s old).")

        raw_markets       = {}
        tag_success_count = 0
        tag_failure_count = 0

        for t_id in CACHED_TAG_IDS:
            tag_markets = gamma_scraper(endpoint="markets", tag_id=t_id, limit=100)

            if tag_markets is None:
                # Hard failure — HTTP request broke
                tag_failure_count += 1

            elif not isinstance(tag_markets, list):
                # FIX: Unexpected non-list response (e.g. API returned a dict).
                # Iterating a dict gives keys, not market objects, so m["id"] would
                # raise TypeError. Count as a failure and skip rather than crashing.
                logger.warning(
                    f"Tag {t_id}: expected list, got {type(tag_markets).__name__}. "
                    f"Skipping to avoid iteration on non-market data."
                )
                tag_failure_count += 1

            else:
                # [] or populated list — both are valid outcomes
                tag_success_count += 1
                for m in tag_markets:
                    if isinstance(m, dict) and "id" in m:
                        raw_markets[m["id"]] = m
                    else:
                        logger.warning(f"Tag {t_id}: skipping malformed market entry: {m!r}")

        if tag_failure_count > 0:
            logger.warning(
                f"Per-tag fetch summary: {tag_success_count} succeeded, "
                f"{tag_failure_count} FAILED (markets from failed tags excluded)."
            )
        else:
            logger.info(f"Per-tag fetch: all {tag_success_count} tag requests succeeded.")

        if CACHED_TAG_IDS and tag_success_count == 0:
            raise RuntimeError(
                "All per-tag market fetches failed; no market data available for ingestion."
            )

        market_list = list(raw_markets.values())
        logger.info(f"Found {len(market_list)} unique active markets across targeted tags.")

        if keyword:
            kw       = keyword.lower()
            filtered = [
                m for m in market_list
                if kw in str(m.get("question",    "")).lower()
                or kw in str(m.get("description", "")).lower()
            ]
        else:
            filtered = market_list

        # FIX: Sort key uses the 3-tuple return from _coerce_numeric_value correctly.
        # [0] extracts the float value; "or 0.0" handles None so sort never crashes.
        filtered.sort(
            key=lambda x: (_coerce_numeric_value(x.get("volume"))[0] or 0.0),
            reverse=True
        )
        if limit_results is not None:
            filtered = filtered[:limit_results]

        rows_saved = save_pipeline_dynamic(conn, filtered, "markets")

        # FIX: Warn if the run took long enough to risk overlapping the next interval
        run_elapsed = time.time() - run_start
        if run_elapsed > RUN_TIME_WARN_SECONDS:
            logger.warning(
                f"[Run #{_run_count}] Slow run: {run_elapsed:.1f}s "
                f"(>{RUN_TIME_WARN_SECONDS}s threshold). "
                f"Consider reducing tag count or increasing INTERVAL_SECONDS."
            )

        logger.info(
            f"[Run #{_run_count}] Complete — {rows_saved} new rows written to DB "
            f"in {run_elapsed:.1f}s."
        )
        _consecutive_failures = 0                                                                                   

    except Exception as e:
        _consecutive_failures += 1
        logger.error(f"[Run #{_run_count}] CRITICAL ERROR: {e}", exc_info=True)

        if _consecutive_failures >= CONSECUTIVE_FAILURE_ALERT_THRESHOLD:
            logger.critical(
                f"ALERT: {_consecutive_failures} consecutive failures. "
                f"Scraper may be stalled — check scraper.log immediately."
            )

        # FIX: Hard stop after MAX_CONSECUTIVE_FAILURES so an external supervisor
        # (systemd, cron, Docker restart policy) can detect the stall and intervene.
        if MAX_CONSECUTIVE_FAILURES and _consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            logger.critical(
                f"Reached {MAX_CONSECUTIVE_FAILURES} consecutive failures. "
                f"Exiting with status 1 for supervisor restart."
            )
            raise SystemExit(1)

    finally:
        if "conn" in locals():
            conn.close()
        log_pythonanywhere_cpu_usage()

def _request_shutdown(signum, frame):
    global _shutdown_requested
    logger.info("Shutdown signal received. Will stop cleanly after current run completes.")
    _shutdown_requested = True


# %%
"""Infinite wall-clock loop with graceful shutdown support."""
_shutdown_requested = False

signal.signal(signal.SIGTERM, _request_shutdown)

if __name__ == "__main__":
    INTERVAL_SECONDS = 300

    logger.info(f"Initializing Wall-Clock Scraper (interval={INTERVAL_SECONDS}s).")
    logger.info("Send SIGTERM or press Ctrl+C to stop gracefully after current run.")

    try:
        while not _shutdown_requested:
            now          = time.time()
            time_to_wait = INTERVAL_SECONDS - (now % INTERVAL_SECONDS)
            next_run     = datetime.fromtimestamp(now + time_to_wait).strftime("%H:%M:%S")
            logger.info(f"Sleeping {int(time_to_wait)}s until next run at {next_run}...")
            time.sleep(time_to_wait)

            if not _shutdown_requested:
                run_job(keyword="gold", limit_results=None)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Scraper stopped cleanly.")

    logger.info(f"Scraper shut down after {_run_count} total runs.")





