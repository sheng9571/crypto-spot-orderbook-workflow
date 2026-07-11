"""Daily data report: scan HF, detect gaps/overlaps, merge, report, cleanup."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


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
# HF Scanning
# ---------------------------------------------------------------------------


def list_hf_files(date: str, config: Config) -> list[HFFile]:
    """List all parquet files for a given date from the HF dataset repo.

    Scans the path prefix '{date}/' and parses each file into an HFFile.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=config.hf_token)
    prefix = f"{date}/"
    files: list[HFFile] = []

    for item in api.list_repo_tree(
        repo_id=config.hf_repo,
        repo_type="dataset",
        path_in_repo=prefix,
        recursive=True,
    ):
        # Only process files (not directories), must be .parquet
        if not hasattr(item, "size") or item.size is None:
            continue
        path = item.rfilename if hasattr(item, "rfilename") else str(item.path)
        if not path.endswith(".parquet"):
            continue
        try:
            hf_file = parse_hf_file(path, item.size)
            files.append(hf_file)
        except (IndexError, ValueError) as e:
            logger.warning("Failed to parse HF file path: %s (%s)", path, e)

    return files


# ---------------------------------------------------------------------------
# Gap Detection
# ---------------------------------------------------------------------------


def detect_gaps(files: list[HFFile], config: Config) -> list[Gap]:
    """Detect missing hours for each exchange/symbol/level combination.

    A gap exists for hour H if no file covers that hour for the given combo.
    """
    # Build a set of (exchange, symbol, level, hour) that exist
    covered: set[tuple[str, str, str, int]] = set()
    for f in files:
        covered.add((f.exchange, f.symbol, f.level, f.hour))

    gaps: list[Gap] = []
    for exchange in config.exchanges:
        for symbol in config.symbols:
            for level in config.levels:
                for hour in range(24):
                    if (exchange, symbol, level, hour) not in covered:
                        gaps.append(Gap(
                            exchange=exchange,
                            symbol=symbol,
                            level=level,
                            hour=hour,
                        ))

    return gaps


# ---------------------------------------------------------------------------
# Overlap Detection
# ---------------------------------------------------------------------------


def detect_overlaps(files: list[HFFile]) -> list[OverlapGroup]:
    """Find groups of files sharing (exchange, symbol, level, hour) with different instance_ids."""
    groups: dict[tuple[str, str, str, int], list[HFFile]] = defaultdict(list)
    for f in files:
        groups[(f.exchange, f.symbol, f.level, f.hour)].append(f)

    overlaps: list[OverlapGroup] = []
    for (exchange, symbol, level, hour), group_files in groups.items():
        instance_ids = {f.instance_id for f in group_files}
        if len(instance_ids) > 1:
            overlaps.append(OverlapGroup(
                exchange=exchange,
                symbol=symbol,
                level=level,
                hour=hour,
                files=group_files,
            ))

    return overlaps


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------


def compute_metrics(files: list[HFFile], gaps: list[Gap]) -> DayMetrics:
    """Compute daily metrics from the file list."""
    if not files:
        return DayMetrics(date="", gaps=gaps)

    date = files[0].date
    total_files = len(files)
    total_bytes = sum(f.size for f in files)

    by_exchange: dict[str, ExchangeMetrics] = {}
    by_symbol: dict[str, SymbolMetrics] = {}
    by_exchange_symbol: dict[str, dict[str, SymbolMetrics]] = defaultdict(dict)

    # Per-exchange
    exchange_files: dict[str, list[HFFile]] = defaultdict(list)
    for f in files:
        exchange_files[f.exchange].append(f)

    # Gaps per exchange
    exchange_gaps: set[str] = {g.exchange for g in gaps}

    for exchange, ex_files in exchange_files.items():
        by_exchange[exchange] = ExchangeMetrics(
            files=len(ex_files),
            bytes=sum(f.size for f in ex_files),
            has_gaps=exchange in exchange_gaps,
        )

    # Per-symbol
    symbol_files: dict[str, list[HFFile]] = defaultdict(list)
    for f in files:
        symbol_files[f.symbol].append(f)

    for symbol, sym_files in symbol_files.items():
        by_symbol[symbol] = SymbolMetrics(
            files=len(sym_files),
            bytes=sum(f.size for f in sym_files),
        )

    # Per-exchange-symbol
    ex_sym_files: dict[tuple[str, str], list[HFFile]] = defaultdict(list)
    for f in files:
        ex_sym_files[(f.exchange, f.symbol)].append(f)

    for (exchange, symbol), es_files in ex_sym_files.items():
        if exchange not in by_exchange_symbol:
            by_exchange_symbol[exchange] = {}
        by_exchange_symbol[exchange][symbol] = SymbolMetrics(
            files=len(es_files),
            bytes=sum(f.size for f in es_files),
        )

    return DayMetrics(
        date=date,
        total_files=total_files,
        total_bytes=total_bytes,
        gaps=gaps,
        by_exchange=by_exchange,
        by_symbol=by_symbol,
        by_exchange_symbol=dict(by_exchange_symbol),
    )


# ---------------------------------------------------------------------------
# Main (placeholder for subsequent tasks)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    config = load_config()
    print(f"Config loaded: {config.hf_repo}, exchanges={config.exchanges}, "
          f"symbols={config.symbols}, levels={config.levels}")
