import json
import logging
import os
import re
import signal
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "Data" / "polymarket_gamma_dynamic.sqlite"
LOG_PATH = BASE_DIR / "logs" / "scraper.log"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

NUMERIC_COLUMNS = {
    "volume", "volumeNum", "liquidity", "liquidityNum",
    "lastTradePrice", "bestBid", "bestAsk",
    "spread", "volume24hr", "volume1wk", "volume1mo",
}
MISSING_NUMERIC_TOKENS = {"", "none", "null", "nan", "n/a", "na", "-", "--", "unknown"}
NUMERIC_SUFFIX_MULTIPLIERS = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}

KEY_FIELDS_TO_MONITOR = {"volume", "lastTradePrice", "bestBid", "bestAsk"}
NULL_RATE_ALERT_THRESHOLD = 0.10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

_consecutive_failures = 0
_run_count = 0
_shutdown_requested = False
CONSECUTIVE_FAILURE_ALERT_THRESHOLD = 3
MAX_CONSECUTIVE_FAILURES = 10
HEARTBEAT_EVERY_N_RUNS = 12
RUN_TIME_WARN_SECONDS = 240

CACHED_TAG_IDS = []
TAG_CACHE_LAST_FETCH = 0.0
TAG_CACHE_TTL = 3600

PYTHONANYWHERE_USERNAME = os.getenv("PYTHONANYWHERE_USERNAME")
PYTHONANYWHERE_API_TOKEN = os.getenv("PYTHONANYWHERE_API_TOKEN")
PYTHONANYWHERE_CPU_TIMEOUT = (5, 15)


def get_retrying_session():
    session = requests.Session()
    session.headers.update({"User-Agent": "PolymarketGoldScraper/1.0"})
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


http_session = get_retrying_session()


def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def log_pythonanywhere_cpu_usage():
    """Fetch and log PythonAnywhere daily CPU usage without raising."""
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
    except Exception as error:
        logger.warning(f"Could not fetch PythonAnywhere CPU usage: {error}")


def _coerce_numeric_value(raw_value):
    if raw_value is None:
        return None, False, False

    if isinstance(raw_value, bool):
        return float(raw_value), False, False

    if isinstance(raw_value, (int, float)):
        try:
            return float(raw_value), False, False
        except (ValueError, TypeError):
            return None, True, False

    text = str(raw_value).strip()
    if text.lower() in MISSING_NUMERIC_TOKENS:
        return None, False, False

    try:
        return float(text), False, False
    except (ValueError, TypeError):
        pass

    normalized = text.replace(",", "").replace("_", "")
    if normalized.startswith("(") and normalized.endswith(")"):
        normalized = "-" + normalized[1:-1].strip()

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", normalized)
    if not match:
        return None, True, True

    try:
        value = float(match.group(0))
    except ValueError:
        return None, True, True

    suffix_match = re.match(r"\s*([kKmMbBtT])", normalized[match.end():])
    if suffix_match:
        value *= NUMERIC_SUFFIX_MULTIPLIERS[suffix_match.group(1).lower()]

    if "%" in normalized:
        value /= 100.0

    return value, False, True


def gamma_scraper(endpoint="markets", fetch_all=True, **kwargs):
    base_url = f"https://gamma-api.polymarket.com/{endpoint}"

    known_keys = {"limit", "offset", "active", "closed"}
    params = {
        "limit": kwargs.get("limit", 100),
        "offset": kwargs.get("offset", 0),
        "active": str(kwargs.get("active", "true")).lower(),
        "closed": str(kwargs.get("closed", "false")).lower(),
    }
    params.update({key: value for key, value in kwargs.items() if key not in known_keys})

    try:
        if not fetch_all:
            response = http_session.get(base_url, params=params, timeout=(5, 30))
            response.raise_for_status()
            return response.json()

        all_items = []
        limit = int(params["limit"])
        offset = int(params["offset"])
        start_offset = offset

        while True:
            current_params = dict(params)
            current_params["limit"] = limit
            current_params["offset"] = offset

            response = http_session.get(base_url, params=current_params, timeout=(5, 30))
            response.raise_for_status()
            page_data = response.json()

            if isinstance(page_data, list):
                page_items = page_data
            elif isinstance(page_data, dict):
                page_items = None
                for key in (endpoint, "data", "results", "items", "rows"):
                    candidate = page_data.get(key)
                    if isinstance(candidate, list):
                        page_items = candidate
                        if offset == start_offset:
                            logger.info(
                                f"/{endpoint}: using wrapped list payload from key '{key}' (API response type=dict)."
                            )
                        break
                if page_items is None:
                    logger.warning(
                        f"/{endpoint}: expected list page, got dict without list payload. Returning as-is (possible API shape change)."
                    )
                    return page_data
            else:
                logger.warning(
                    f"/{endpoint}: expected list page, got {type(page_data).__name__}. Returning as-is (single-object response or API shape change)."
                )
                return page_data

            all_items.extend(page_items)
            if len(page_items) < limit:
                break
            offset += limit

        return all_items

    except Exception as error:
        logger.error(f"Error fetching /{endpoint}: {error}")
        return None


def save_pipeline_dynamic(conn, data, table_name="markets"):
    if not data:
        return 0

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    for item in data:
        item["scraped_at"] = current_time

    cursor = conn.cursor()
    all_keys = set()
    for item in data:
        all_keys.update(item.keys())

    columns = sorted(list(all_keys))
    if "scraped_at" in columns:
        columns.remove("scraped_at")
    if "id" in columns:
        columns.remove("id")
    columns.insert(0, "id")
    columns.insert(1, "scraped_at")

    def _col_type(column):
        return "REAL" if column in NUMERIC_COLUMNS else "TEXT"

    col_defs = [f'"{column}" {_col_type(column)}' for column in columns]
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
    for column in columns:
        if column not in existing_columns:
            cursor.execute(f'ALTER TABLE {table_name} ADD COLUMN "{column}" {_col_type(column)}')

    coerce_failures = 0
    coerce_failures_by_col = {}
    relaxed_parse_count = 0
    rows_to_insert = []

    for item in data:
        row = []
        for column in columns:
            value = item.get(column)
            if column in NUMERIC_COLUMNS:
                coerced_val, failed, used_relaxed = _coerce_numeric_value(value)
                row.append(coerced_val)
                if failed:
                    coerce_failures += 1
                    coerce_failures_by_col[column] = coerce_failures_by_col.get(column, 0) + 1
                elif used_relaxed:
                    relaxed_parse_count += 1
            elif isinstance(value, (dict, list, bool)):
                row.append(json.dumps(value))
            else:
                row.append(value)
        rows_to_insert.append(tuple(row))

    if relaxed_parse_count > 0:
        logger.info(f"[{table_name}] Relaxed numeric parsing recovered {relaxed_parse_count} value(s).")

    if coerce_failures > 0:
        by_col = ", ".join(
            f"{column}={count}"
            for column, count in sorted(coerce_failures_by_col.items(), key=lambda item: (-item[1], item[0]))
        )
        logger.warning(
            f"[{table_name}] {coerce_failures} numeric coercion failure(s) - stored as NULL."
            + (f" Breakdown: {by_col}" if by_col else "")
        )

    if table_name == "markets":
        row_count = len(rows_to_insert)
        for field in KEY_FIELDS_TO_MONITOR:
            if field in columns:
                field_idx = columns.index(field)
                null_count = sum(1 for row in rows_to_insert if row[field_idx] is None)
                null_rate = null_count / row_count
                if null_rate > NULL_RATE_ALERT_THRESHOLD:
                    logger.warning(
                        f"[{table_name}] High null rate on '{field}': {null_rate:.0%} ({null_count}/{row_count} rows) - likely genuine API absence; filter at modeling time."
                    )

    insert_sql = (
        f'INSERT OR IGNORE INTO {table_name} '
        f'({", ".join(f"{chr(34)}{column}{chr(34)}" for column in columns)}) '
        f'VALUES ({", ".join(["?"] * len(columns))})'
    )

    for attempt in range(3):
        try:
            cursor.executemany(insert_sql, rows_to_insert)
            conn.commit()
            break
        except sqlite3.OperationalError as error:
            if attempt < 2:
                wait = 2 ** attempt
                logger.warning(f"[{table_name}] DB write error (attempt {attempt + 1}/3): {error}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"[{table_name}] DB write failed after 3 attempts.", exc_info=True)
                raise

    rows_inserted = cursor.rowcount
    rows_ignored = len(rows_to_insert) - rows_inserted
    if rows_ignored > 0:
        logger.info(f"[{table_name}] {rows_inserted} new rows inserted, {rows_ignored} duplicate rows ignored.")

    return rows_inserted


def run_job(keyword="gold", limit_results=None):
    global CACHED_TAG_IDS, TAG_CACHE_LAST_FETCH, _consecutive_failures, _run_count

    _run_count += 1
    limit_text = "All" if limit_results is None else limit_results
    logger.info(f"[Run #{_run_count}] Starting scrape (keyword='{keyword}', limit={limit_text})...")

    run_start = time.time()

    if _run_count % HEARTBEAT_EVERY_N_RUNS == 0:
        logger.info(
            f"[HEARTBEAT] Run #{_run_count} - scraper alive. Consecutive failures: {_consecutive_failures}."
        )

    try:
        conn = init_db()

        cache_age = time.time() - TAG_CACHE_LAST_FETCH
        if not CACHED_TAG_IDS or cache_age > TAG_CACHE_TTL:
            logger.info(f"Tag cache stale ({int(cache_age)}s old). Fetching from API...")
            all_tags = gamma_scraper(endpoint="tags", limit=100)
            if all_tags is None:
                raise RuntimeError("Tag fetch returned None (request failed). Cannot proceed.")
            if not isinstance(all_tags, list):
                raise RuntimeError(
                    f"Tag fetch returned unexpected type {type(all_tags).__name__} (expected list). Cannot build tag cache."
                )
            target_keywords = ["finance", "crypto", "geopolitics", "politics"]
            CACHED_TAG_IDS = list({
                tag.get("id")
                for tag in all_tags
                if isinstance(tag, dict) and any(
                    keyword_part in str(tag.get("label", "")).lower()
                    or keyword_part in str(tag.get("slug", "")).lower()
                    for keyword_part in target_keywords
                )
            })
            TAG_CACHE_LAST_FETCH = time.time()
            save_pipeline_dynamic(conn, all_tags, "tags")
            logger.info(f"Cached {len(CACHED_TAG_IDS)} tag IDs for the next {TAG_CACHE_TTL}s.")
        else:
            logger.info(f"Using {len(CACHED_TAG_IDS)} cached tag IDs ({int(cache_age)}s old).")

        raw_markets = {}
        tag_success_count = 0
        tag_failure_count = 0

        for tag_id in CACHED_TAG_IDS:
            tag_markets = gamma_scraper(endpoint="markets", tag_id=tag_id, limit=100)

            if tag_markets is None:
                tag_failure_count += 1
            elif not isinstance(tag_markets, list):
                logger.warning(
                    f"Tag {tag_id}: expected list, got {type(tag_markets).__name__}. Skipping to avoid iteration on non-market data."
                )
                tag_failure_count += 1
            else:
                tag_success_count += 1
                for market in tag_markets:
                    if isinstance(market, dict) and "id" in market:
                        raw_markets[market["id"]] = market
                    else:
                        logger.warning(f"Tag {tag_id}: skipping malformed market entry: {market!r}")

        if tag_failure_count > 0:
            logger.warning(
                f"Per-tag fetch summary: {tag_success_count} succeeded, {tag_failure_count} FAILED (markets from failed tags excluded)."
            )
        else:
            logger.info(f"Per-tag fetch: all {tag_success_count} tag requests succeeded.")

        if CACHED_TAG_IDS and tag_success_count == 0:
            raise RuntimeError("All per-tag market fetches failed; no market data available for ingestion.")

        market_list = list(raw_markets.values())
        logger.info(f"Found {len(market_list)} unique active markets across targeted tags.")

        if keyword:
            keyword_lower = keyword.lower()
            filtered = [
                market for market in market_list
                if keyword_lower in str(market.get("question", "")).lower()
                or keyword_lower in str(market.get("description", "")).lower()
            ]
        else:
            filtered = market_list

        filtered.sort(
            key=lambda market: (_coerce_numeric_value(market.get("volume"))[0] or 0.0),
            reverse=True,
        )
        if limit_results is not None:
            filtered = filtered[:limit_results]

        rows_saved = save_pipeline_dynamic(conn, filtered, "markets")

        run_elapsed = time.time() - run_start
        if run_elapsed > RUN_TIME_WARN_SECONDS:
            logger.warning(
                f"[Run #{_run_count}] Slow run: {run_elapsed:.1f}s (>{RUN_TIME_WARN_SECONDS}s threshold). Consider reducing tag count or increasing INTERVAL_SECONDS."
            )

        logger.info(
            f"[Run #{_run_count}] Complete - {rows_saved} new rows written to DB in {run_elapsed:.1f}s."
        )
        _consecutive_failures = 0

    except Exception as error:
        _consecutive_failures += 1
        logger.error(f"[Run #{_run_count}] CRITICAL ERROR: {error}", exc_info=True)

        if _consecutive_failures >= CONSECUTIVE_FAILURE_ALERT_THRESHOLD:
            logger.critical(
                f"ALERT: {_consecutive_failures} consecutive failures. Scraper may be stalled - check scraper.log immediately."
            )

        if MAX_CONSECUTIVE_FAILURES and _consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            logger.critical(
                f"Reached {MAX_CONSECUTIVE_FAILURES} consecutive failures. Exiting with status 1 for supervisor restart."
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