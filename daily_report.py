"""Daily data report: scan HF, detect gaps/overlaps, merge, report, cleanup."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

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
    hf_token = os.environ.get("HF_TOKEN", "")
    hf_repo = os.environ.get("HF_REPO_NAME", "")
    if not hf_token or not hf_repo:
        raise SystemExit("ERROR: HF_TOKEN and HF_REPO_NAME environment variables are required")

    return Config(
        hf_token=hf_token,
        hf_repo=hf_repo,
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
    Returns empty list if the date directory doesn't exist on HF.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import EntryNotFoundError

    api = HfApi(token=config.hf_token)
    prefix = f"{date}/"
    files: list[HFFile] = []

    try:
        tree_iter = api.list_repo_tree(
            repo_id=config.hf_repo,
            repo_type="dataset",
            path_in_repo=prefix,
            recursive=True,
        )
        for item in tree_iter:
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
    except EntryNotFoundError:
        logger.info("No data directory found for %s (404), treating as 0 files", date)

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


def compute_metrics(files: list[HFFile], gaps: list[Gap], date: str = "") -> DayMetrics:
    """Compute daily metrics from the file list."""
    if not files:
        return DayMetrics(date=date, gaps=gaps)

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
# Overlap Merge
# ---------------------------------------------------------------------------


def merge_overlap(group: OverlapGroup, config: Config) -> MergeResult | None:
    """Download overlapping parquet files, merge, deduplicate, upload, delete originals.

    Returns MergeResult on success, None on failure.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    api = HfApi(token=config.hf_token)

    try:
        # Download all files in the overlap group
        tables = []
        for f in group.files:
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                api.hf_hub_download(
                    repo_id=config.hf_repo,
                    repo_type="dataset",
                    filename=f.path,
                    local_dir=tempfile.gettempdir(),
                    local_dir_use_symlinks=False,
                )
                # hf_hub_download returns a path, but let's use the direct download
                downloaded = api.hf_hub_download(
                    repo_id=config.hf_repo,
                    repo_type="dataset",
                    filename=f.path,
                )
                table = pq.read_table(downloaded)
                tables.append(table)
            except Exception as e:
                logger.error("Failed to download %s: %s", f.path, e)
                return None

        if not tables:
            return None

        import pyarrow as pa

        # Concatenate all tables
        combined = pa.concat_tables(tables)
        rows_before = combined.num_rows

        # Sort by local_ts
        indices = combined.column("local_ts").to_pylist()
        sort_indices = sorted(range(len(indices)), key=lambda i: indices[i])
        combined = combined.take(sort_indices)

        # Deduplicate by key keeping first occurrence (already sorted by local_ts)
        # L2 has sequence column -> use (exchange, symbol, sequence) as dedup key
        # L1 has no sequence -> use (exchange, symbol, local_ts) as dedup key
        col_names = combined.column_names
        if "sequence" in col_names:
            dedup_cols = ["exchange", "symbol", "sequence"]
        else:
            dedup_cols = ["exchange", "symbol", "local_ts"]

        seen: set[tuple] = set()
        keep_indices: list[int] = []
        for i in range(combined.num_rows):
            key = tuple(combined.column(c)[i].as_py() for c in dedup_cols)
            if key not in seen:
                seen.add(key)
                keep_indices.append(i)

        combined = combined.take(keep_indices)
        rows_after = combined.num_rows

        # Write merged file with instance_id="merged"
        merged_filename = (
            f"{group.exchange}_{group.symbol}_{group.level}_"
            f"{group.files[0].date}_{group.files[0].start_ts}_merged.parquet"
        )
        merged_remote_path = (
            f"{group.files[0].date}/{group.exchange}/{group.symbol}/"
            f"{group.level}/{merged_filename}"
        )

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            merged_local = tmp.name

        pq.write_table(combined, merged_local, compression="zstd")

        # Upload merged file
        api.upload_file(
            path_or_fileobj=merged_local,
            path_in_repo=merged_remote_path,
            repo_id=config.hf_repo,
            repo_type="dataset",
        )

        # Delete originals
        for f in group.files:
            try:
                api.delete_file(
                    path_in_repo=f.path,
                    repo_id=config.hf_repo,
                    repo_type="dataset",
                )
            except Exception as e:
                logger.warning("Failed to delete original %s: %s", f.path, e)

        # Clean up temp file
        Path(merged_local).unlink(missing_ok=True)

        # Compute overlap duration
        timestamps = [f.start_ts for f in group.files]
        overlap_duration = max(timestamps) - min(timestamps)

        return MergeResult(
            exchange=group.exchange,
            symbol=group.symbol,
            level=group.level,
            hour=group.hour,
            files_merged=[f.path for f in group.files],
            rows_before=rows_before,
            rows_after=rows_after,
            overlap_duration_seconds=overlap_duration,
        )

    except Exception as e:
        logger.error("Merge failed for %s/%s/%s hour %d: %s",
                     group.exchange, group.symbol, group.level, group.hour, e)
        return None


# ---------------------------------------------------------------------------
# History JSON
# ---------------------------------------------------------------------------


def load_history(history_path: str = "reports/history.json") -> dict:
    """Load history JSON, initialize if missing or malformed."""
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        # Validate structure
        if "daily" not in history or "totals" not in history:
            raise ValueError("Invalid history structure")
        return history
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return {
            "daily": [],
            "totals": {
                "all_time_bytes": 0,
                "all_time_files": 0,
                "first_date": None,
                "last_date": None,
            },
        }


def update_history(history: dict, metrics: DayMetrics) -> dict:
    """Append today's metrics to history and update totals.

    If an entry for the same date already exists, replace it (idempotent re-run).
    """
    day_entry = {
        "date": metrics.date,
        "total_files": metrics.total_files,
        "total_bytes": metrics.total_bytes,
        "gaps": [
            {"exchange": g.exchange, "symbol": g.symbol, "level": g.level, "hour": g.hour}
            for g in metrics.gaps
        ],
        "by_exchange": {
            k: {"files": v.files, "bytes": v.bytes, "has_gaps": v.has_gaps}
            for k, v in metrics.by_exchange.items()
        },
        "by_symbol": {
            k: {"files": v.files, "bytes": v.bytes}
            for k, v in metrics.by_symbol.items()
        },
        "by_exchange_symbol": {
            ex: {sym: {"files": sm.files, "bytes": sm.bytes} for sym, sm in syms.items()}
            for ex, syms in metrics.by_exchange_symbol.items()
        },
    }

    # Check for existing entry with same date (idempotent re-run)
    existing_idx = None
    for i, entry in enumerate(history["daily"]):
        if entry["date"] == metrics.date:
            existing_idx = i
            break

    totals = history["totals"]

    if existing_idx is not None:
        # Replace existing entry, adjust totals (subtract old, add new)
        old_entry = history["daily"][existing_idx]
        totals["all_time_bytes"] -= old_entry["total_bytes"]
        totals["all_time_files"] -= old_entry["total_files"]
        history["daily"][existing_idx] = day_entry
    else:
        history["daily"].append(day_entry)

    # Update totals
    totals["all_time_bytes"] += metrics.total_bytes
    totals["all_time_files"] += metrics.total_files
    totals["last_date"] = metrics.date
    if totals["first_date"] is None:
        totals["first_date"] = metrics.date

    return history


def save_history(history: dict, history_path: str = "reports/history.json") -> None:
    """Write history JSON to disk."""
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def compute_rolling_totals(history: dict, days: int) -> int:
    """Sum total_bytes from the last N daily entries."""
    entries = history["daily"]
    recent = entries[-days:] if len(entries) >= days else entries
    return sum(e["total_bytes"] for e in recent)


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------


def _format_bytes(n: int) -> str:
    """Format bytes into human-readable string."""
    if n < 1024:
        return f"{n} B"
    elif n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    elif n < 1024 ** 3:
        return f"{n / 1024 ** 2:.1f} MB"
    else:
        return f"{n / 1024 ** 3:.2f} GB"


def generate_report_md(
    metrics: DayMetrics,
    history: dict,
    merges: list[MergeResult],
) -> str:
    """Generate a daily report markdown string."""
    lines: list[str] = []

    # Header
    lines.append(f"# Daily Report: {metrics.date}")
    lines.append("")

    # Summary
    rolling_7d = compute_rolling_totals(history, 7)
    rolling_30d = compute_rolling_totals(history, 30)
    all_time = history["totals"]["all_time_bytes"]

    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Files today | {metrics.total_files} |")
    lines.append(f"| Gaps detected | {len(metrics.gaps)} |")
    lines.append(f"| Size today | {_format_bytes(metrics.total_bytes)} |")
    lines.append(f"| Size (7-day) | {_format_bytes(rolling_7d)} |")
    lines.append(f"| Size (30-day) | {_format_bytes(rolling_30d)} |")
    lines.append(f"| Size (all-time) | {_format_bytes(all_time)} |")
    lines.append("")

    # By Exchange
    lines.append("## By Exchange")
    lines.append("")
    lines.append("| Exchange | Files | Size | Gaps |")
    lines.append("|----------|-------|------|------|")
    for exchange, em in sorted(metrics.by_exchange.items()):
        gap_status = "⚠️ Yes" if em.has_gaps else "✅ No"
        lines.append(f"| {exchange} | {em.files} | {_format_bytes(em.bytes)} | {gap_status} |")
    lines.append("")

    # By Symbol
    lines.append("## By Symbol")
    lines.append("")
    lines.append("| Symbol | Files | Size |")
    lines.append("|--------|-------|------|")
    for symbol, sm in sorted(metrics.by_symbol.items()):
        lines.append(f"| {symbol} | {sm.files} | {_format_bytes(sm.bytes)} |")
    lines.append("")

    # Gaps Detected
    lines.append("## Gaps Detected")
    lines.append("")
    if metrics.gaps:
        lines.append("| Exchange | Symbol | Level | Hour (UTC) |")
        lines.append("|----------|--------|-------|------------|")
        for g in sorted(metrics.gaps, key=lambda x: (x.exchange, x.symbol, x.level, x.hour)):
            lines.append(f"| {g.exchange} | {g.symbol} | {g.level} | {g.hour:02d}:00 |")
    else:
        lines.append("No gaps detected. ✅")
    lines.append("")

    # Merges
    if merges:
        lines.append("## Merges Performed")
        lines.append("")
        for m in merges:
            lines.append(f"### {m.exchange}/{m.symbol}/{m.level} hour {m.hour:02d}")
            lines.append(f"- Files merged: {len(m.files_merged)}")
            lines.append(f"- Rows before: {m.rows_before} → after: {m.rows_after}")
            lines.append(f"- Overlap duration: {m.overlap_duration_seconds}s")
            lines.append("")

    return "\n".join(lines)


def save_report(report_md: str, date: str) -> str:
    """Save report markdown to reports/daily/{date}.md. Returns the file path."""
    report_path = f"reports/daily/{date}.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    return report_path


# ---------------------------------------------------------------------------
# Turso Heartbeat Cleanup
# ---------------------------------------------------------------------------


def cleanup_heartbeats(turso_url: str, turso_token: str, threshold_days: int = 7) -> int:
    """Delete stale heartbeat records from Turso.

    Deletes records where status='stopped' AND updated_at is older than threshold_days.
    Returns number of rows deleted. Returns 0 on failure (non-critical).
    """
    import requests

    if not turso_url or not turso_token:
        logger.info("Turso not configured, skipping heartbeat cleanup")
        return 0

    sql = (
        "DELETE FROM heartbeats "
        "WHERE status = 'stopped' "
        f"AND updated_at < datetime('now', '-{threshold_days} days')"
    )

    try:
        resp = requests.post(
            f"{turso_url}/v2/pipeline",
            headers={
                "Authorization": f"Bearer {turso_token}",
                "Content-Type": "application/json",
            },
            json={
                "requests": [
                    {"type": "execute", "stmt": {"sql": sql}},
                    {"type": "close"},
                ]
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        # Extract affected row count from response
        results = data.get("results", [])
        if results and "response" in results[0]:
            affected = results[0]["response"].get("result", {}).get("affected_row_count", 0)
            logger.info("Turso cleanup: deleted %d stale heartbeat records", affected)
            return affected
        return 0
    except Exception as e:
        logger.warning("Turso cleanup failed (non-critical): %s", e)
        return 0


# ---------------------------------------------------------------------------
# Git Commit & Push
# ---------------------------------------------------------------------------


def git_commit_and_push(files: list[str], date: str) -> None:
    """Stage, commit, and push report files. Skips if nothing changed."""
    import subprocess

    try:
        subprocess.run(["git", "add"] + files, check=True)

        # Check if there are staged changes
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            capture_output=True,
        )
        if result.returncode == 0:
            logger.info("No changes to commit, skipping git push")
            return

        subprocess.run(
            ["git", "commit", "-m", f"chore: daily report {date}"],
            check=True,
        )
        subprocess.run(["git", "push"], check=True)
        logger.info("Git push successful")
    except subprocess.CalledProcessError as e:
        logger.error("Git operation failed: %s", e)
        raise


# ---------------------------------------------------------------------------
# HF Date Discovery
# ---------------------------------------------------------------------------


def list_hf_dates(config: Config) -> list[str]:
    """List all date directories in the HF repo root (e.g. ['2026-07-10', '2026-07-11']).

    Only returns entries that look like YYYY-MM-DD date folders.
    """
    import re

    from huggingface_hub import HfApi

    api = HfApi(token=config.hf_token)
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    dates: list[str] = []

    for item in api.list_repo_tree(
        repo_id=config.hf_repo,
        repo_type="dataset",
        path_in_repo="",
        recursive=False,
    ):
        # Directories have no size attribute or size is None
        name = item.rfilename if hasattr(item, "rfilename") else str(item.path)
        # Remove trailing slash if present
        name = name.strip("/")
        if date_pattern.match(name):
            dates.append(name)

    return sorted(dates)


# ---------------------------------------------------------------------------
# Process Single Date
# ---------------------------------------------------------------------------


def process_date(date: str, config: Config, history: dict) -> tuple[str, DayMetrics]:
    """Process a single date: merge overlaps, detect gaps, compute metrics, save report.

    Returns (report_path, metrics).
    """
    # 1. List HF files
    files = list_hf_files(date, config)
    logger.info("[%s] Found %d files", date, len(files))

    # 2. Detect and merge overlaps
    overlaps = detect_overlaps(files)
    merges: list[MergeResult] = []
    if overlaps:
        logger.info("[%s] Detected %d overlap groups, merging...", date, len(overlaps))
        for group in overlaps:
            result = merge_overlap(group, config)
            if result:
                merges.append(result)
        # Re-scan after merge
        files = list_hf_files(date, config)

    # 3. Detect gaps and compute metrics
    gaps = detect_gaps(files, config)
    metrics = compute_metrics(files, gaps, date=date)
    metrics.merges = merges

    # 4. Update history (in-place)
    update_history(history, metrics)

    # 5. Generate and save report
    report_md = generate_report_md(metrics, history, merges)
    report_path = save_report(report_md, date)

    return report_path, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Orchestrate: find missing dates -> process each -> cleanup -> commit."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config()

    # Determine which dates to process
    force_date = os.environ.get("REPORT_DATE", "").strip()

    if force_date:
        # Manual mode: process only the specified date
        dates_to_process = [force_date]
        logger.info("Manual mode: processing date %s", force_date)
    else:
        # Auto mode: find all HF dates not yet in history
        from datetime import timedelta

        all_hf_dates = list_hf_dates(config)
        logger.info("Found %d date folders on HF", len(all_hf_dates))

        history = load_history()
        already_done = {entry["date"] for entry in history["daily"]}

        # Only process dates up to yesterday UTC (today may still be incomplete)
        yesterday = (datetime.now(tz=timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        dates_to_process = [
            d for d in all_hf_dates
            if d not in already_done and d <= yesterday
        ]

        if not dates_to_process:
            logger.info("All dates up to %s already processed, nothing to do", yesterday)

    # Load history (may already be loaded above, but safe to reload)
    history = load_history()

    # Process each date
    report_paths: list[str] = []
    for date in sorted(dates_to_process):
        logger.info("Processing %s...", date)
        report_path, _ = process_date(date, config, history)
        report_paths.append(report_path)

    # Save history once (all dates accumulated)
    if report_paths:
        save_history(history)

    # Turso heartbeat cleanup (non-critical, run once per workflow)
    deleted = cleanup_heartbeats(
        config.turso_url, config.turso_token, config.cleanup_threshold_days
    )
    if deleted:
        logger.info("Cleaned up %d stale heartbeat records", deleted)

    # Git commit and push all reports at once
    if report_paths:
        all_files = report_paths + ["reports/history.json"]
        dates_str = f"{dates_to_process[0]}" if len(dates_to_process) == 1 else f"{dates_to_process[0]}..{dates_to_process[-1]}"
        git_commit_and_push(all_files, dates_str)

    logger.info("Daily report workflow complete")


if __name__ == "__main__":
    main()
