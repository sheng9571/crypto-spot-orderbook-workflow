"""Daily data report: scan HF, detect gaps/overlaps, merge, report, cleanup."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class Config:
    hf_token: str
    hf_repo: str
    turso_url: str
    turso_token: str
    exchanges: list[str]
    symbols: list[str]
    levels: list[str]
    cleanup_threshold_days: int = 7


@dataclass
class HFFile:
    path: str
    size: int
    exchange: str
    symbol: str
    level: str
    date: str
    hour: int
    instance_id: str
    start_ts: int


@dataclass
class OverlapGroup:
    exchange: str
    symbol: str
    level: str
    hour: int
    files: list[HFFile]


@dataclass
class Gap:
    exchange: str
    symbol: str
    level: str
    hour: int



@dataclass
class ExchangeMetrics:
    files: int = 0
    bytes: int = 0
    has_gaps: bool = False


@dataclass
class SymbolMetrics:
    files: int = 0
    bytes: int = 0


@dataclass
class MergeResult:
    exchange: str
    symbol: str
    level: str
    hour: int
    files_merged: list[str]
    rows_before: int
    rows_after: int
    overlap_duration_seconds: int


@dataclass
class DayMetrics:
    date: str
    total_files: int = 0
    total_bytes: int = 0
    gaps: list[Gap] = field(default_factory=list)
    by_exchange: dict[str, ExchangeMetrics] = field(default_factory=dict)
    by_symbol: dict[str, SymbolMetrics] = field(default_factory=dict)
    by_exchange_symbol: dict[str, dict[str, SymbolMetrics]] = field(default_factory=dict)
    merges: list[MergeResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------


def load_config() -> Config:
    """Load configuration from environment variables."""
    return Config(
        hf_token=os.environ["HF_TOKEN"],
        hf_repo=os.environ["HF_REPO_NAME"],
        turso_url=os.environ.get("TURSO_URL", ""),
        turso_token=os.environ.get("TURSO_TOKEN", ""),
        exchanges=[
            e.strip()
            for e in os.environ.get("EXCHANGES", "binance").split(",")
            if e.strip()
        ],
        symbols=[
            s.strip()
            for s in os.environ.get("SYMBOLS", "btcusdt").split(",")
            if s.strip()
        ],
        levels=[
            l.strip()
            for l in os.environ.get("LEVELS", "l1,l2").split(",")
            if l.strip()
        ],
        cleanup_threshold_days=int(
            os.environ.get("CLEANUP_THRESHOLD_DAYS", "7")
        ),
    )


# ---------------------------------------------------------------------------
# HF File Parsing
# ---------------------------------------------------------------------------


def parse_hf_file(path: str, size: int) -> HFFile:
    """Parse an HF file path into an HFFile dataclass.

    Expected path format:
        {date}/{exchange}/{symbol}/{level}/{exchange}_{symbol}_{level}_{date}_{start_ts}_{instance_id}.parquet
    Example:
        2026-07-10/binance/btcusdt/l2/binance_btcusdt_l2_2026-07-10_1783689306_0jekh1.parquet
    """
    parts = path.split("/")
    # parts: [date, exchange, symbol, level, filename]
    date = parts[0]
    exchange = parts[1]
    symbol = parts[2]
    level = parts[3]
    filename = parts[4]

    # Parse filename: {exchange}_{symbol}_{level}_{date}_{start_ts}_{instance_id}.parquet
    name_no_ext = filename.removesuffix(".parquet")
    # Split from the right to handle potential underscores in earlier fields
    # Format: exchange_symbol_level_YYYY-MM-DD_startts_instanceid
    # Date contains a hyphen pattern, split by underscore
    segments = name_no_ext.split("_")
    # segments: [exchange, symbol, level, YYYY, MM, DD, start_ts, instance_id]
    # Wait - date is YYYY-MM-DD which contains hyphens not underscores
    # So: binance_btcusdt_l2_2026-07-10_1783689306_0jekh1
    # Split by _: ['binance', 'btcusdt', 'l2', '2026-07-10', '1783689306', '0jekh1']
    start_ts = int(segments[-2])
    instance_id = segments[-1]

    # Derive hour from start_ts (seconds since epoch, UTC)
    hour = datetime.fromtimestamp(start_ts, tz=timezone.utc).hour

    return HFFile(
        path=path,
        size=size,
        exchange=exchange,
        symbol=symbol,
        level=level,
        date=date,
        hour=hour,
        instance_id=instance_id,
        start_ts=start_ts,
    )


# ---------------------------------------------------------------------------
# Main (placeholder for subsequent tasks)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    config = load_config()
    print(f"Config loaded: {config.hf_repo}, exchanges={config.exchanges}, "
          f"symbols={config.symbols}, levels={config.levels}")
