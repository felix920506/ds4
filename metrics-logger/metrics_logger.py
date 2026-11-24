#!/usr/bin/env python3
"""Scrape ds4-server's GET /metrics and log every sample into SQLite.

ds4-server exports cumulative counters, histograms and gauges in Prometheus
text format (docs/SERVER.md, "Metrics").  This tool polls one or more servers
on a fixed interval and appends each scrape to a SQLite database, so a
deployment's history can be queried later without running Prometheus.

Schema:

  scrapes   one row per poll: target URL, unix time, success, scrape latency,
            error text, and the server's reported uptime (used to detect
            restarts, which reset every counter).
  metrics   HELP/TYPE metadata per sample name.  Histogram families are
            recorded under their _bucket, _sum and _count sample names.
  samples   (scrape_id, name, labels, value).  labels is the canonical
            Prometheus label set, keys sorted, e.g. endpoint="chat"; '' when
            the sample has none.
  sample_deltas
            view: each sample next to the previous successful scrape of the
            same series on the same target, with delta and per-second rate.
            Counter deltas are NULL across a server restart.

Example queries:

  -- decode throughput per interval
  SELECT datetime(ts, 'unixepoch'), rate FROM sample_deltas
   WHERE name = 'ds4_decode_tokens_total' ORDER BY ts;

  -- latest queue depth per target
  SELECT target, value FROM sample_deltas
   WHERE name = 'ds4_server_queued_requests'
   GROUP BY target HAVING ts = MAX(ts);

Only the standard library is used.  Settings can come from flags or from
DS4_METRICS_* environment variables (see --help), which is how the container
image is configured.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import signal
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

log = logging.getLogger("ds4-metrics-logger")

UPTIME_METRIC = "ds4_server_uptime_seconds"

SCHEMA = """
CREATE TABLE IF NOT EXISTS scrapes (
    id          INTEGER PRIMARY KEY,
    target      TEXT    NOT NULL,
    ts          REAL    NOT NULL,
    ok          INTEGER NOT NULL,
    duration_s  REAL    NOT NULL,
    error       TEXT,
    uptime_s    REAL
);
CREATE INDEX IF NOT EXISTS scrapes_target_ts ON scrapes(target, ts);
CREATE INDEX IF NOT EXISTS scrapes_ts ON scrapes(ts);

CREATE TABLE IF NOT EXISTS metrics (
    name  TEXT PRIMARY KEY,
    type  TEXT NOT NULL,
    help  TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS samples (
    scrape_id  INTEGER NOT NULL REFERENCES scrapes(id) ON DELETE CASCADE,
    name       TEXT    NOT NULL,
    labels     TEXT    NOT NULL,
    value      REAL,
    PRIMARY KEY (scrape_id, name, labels)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS samples_series ON samples(name, labels, scrape_id);

DROP VIEW IF EXISTS sample_deltas;
CREATE VIEW sample_deltas AS
WITH ordered AS (
    SELECT sc.target, sc.ts, sc.uptime_s, s.scrape_id, s.name, s.labels, s.value,
           COALESCE(m.type, 'untyped') AS type,
           LAG(sc.ts)       OVER w AS prev_ts,
           LAG(sc.uptime_s) OVER w AS prev_uptime_s,
           LAG(s.value)     OVER w AS prev_value
      FROM samples s
      JOIN scrapes sc ON sc.id = s.scrape_id
      LEFT JOIN metrics m ON m.name = s.name
    WINDOW w AS (PARTITION BY sc.target, s.name, s.labels ORDER BY sc.ts)
), classified AS (
    SELECT *,
           type IN ('counter', 'histogram') AS cumulative,
           (prev_value IS NOT NULL
            AND ((uptime_s IS NOT NULL AND prev_uptime_s IS NOT NULL
                  AND uptime_s < prev_uptime_s)
                 OR value < prev_value)) AS reset
      FROM ordered
)
SELECT target, ts, scrape_id, name, labels, type, value,
       ts - prev_ts AS interval_s,
       CASE WHEN cumulative AND reset THEN NULL ELSE value - prev_value END AS delta,
       CASE WHEN NOT cumulative OR reset OR ts <= prev_ts THEN NULL
            ELSE (value - prev_value) / (ts - prev_ts) END AS rate
  FROM classified;
"""


# ---------------------------------------------------------------------------
# Prometheus text exposition format parsing.

@dataclass
class Sample:
    name: str
    labels: str
    value: float


@dataclass
class Exposition:
    samples: list[Sample] = field(default_factory=list)
    types: dict[str, str] = field(default_factory=dict)  # family -> type
    helps: dict[str, str] = field(default_factory=dict)  # family -> help

    def sample_metadata(self) -> dict[str, tuple[str, str]]:
        """Map each sample name present to (type, help) of its family."""
        out: dict[str, tuple[str, str]] = {}
        for s in self.samples:
            if s.name in out:
                continue
            family = s.name
            if family not in self.types:
                for suffix in ("_bucket", "_sum", "_count"):
                    base = s.name[: -len(suffix)]
                    if s.name.endswith(suffix) and self.types.get(base) in ("histogram", "summary"):
                        family = base
                        break
            out[s.name] = (self.types.get(family, "untyped"), self.helps.get(family, ""))
        return out

    def value(self, name: str, labels: str = "") -> float | None:
        for s in self.samples:
            if s.name == name and s.labels == labels:
                return s.value
        return None


_NAME = r"[a-zA-Z_:][a-zA-Z0-9_:]*"
_SAMPLE_RE = re.compile(rf"^({_NAME})(?:\{{(.*)\}})?\s+(\S+)(?:\s+-?\d+)?\s*$")
_LABEL_RE = re.compile(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"\s*(?:,|$)')
_META_RE = re.compile(rf"^#\s+(HELP|TYPE)\s+({_NAME})(?:\s+(.*))?$")


def _parse_value(text: str) -> float:
    lowered = text.lower()
    if lowered in ("+inf", "inf"):
        return math.inf
    if lowered == "-inf":
        return -math.inf
    return float(text)  # also handles NaN


def _canonical_labels(body: str | None) -> str:
    if not body or not body.strip():
        return ""
    pairs = []
    pos = 0
    while pos < len(body):
        m = _LABEL_RE.match(body, pos)
        if not m:
            raise ValueError(f"bad label set {{{body}}}")
        pairs.append((m.group(1), m.group(2)))
        pos = m.end()
    return ",".join(f'{k}="{v}"' for k, v in sorted(pairs))


def parse_exposition(text: str) -> Exposition:
    exp = Exposition()
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = _META_RE.match(line)
            if m:
                kind, name, rest = m.group(1), m.group(2), (m.group(3) or "").strip()
                if kind == "TYPE":
                    exp.types[name] = rest or "untyped"
                else:
                    exp.helps[name] = rest
            continue
        m = _SAMPLE_RE.match(line)
        if not m:
            log.warning("skipping unparsable line %d: %r", lineno, raw)
            continue
        try:
            exp.samples.append(Sample(m.group(1), _canonical_labels(m.group(2)),
                                      _parse_value(m.group(3))))
        except ValueError as e:
            log.warning("skipping line %d: %s", lineno, e)
    return exp


# ---------------------------------------------------------------------------
# Storage.

def open_db(path: str) -> sqlite3.Connection:
    if path != ":memory:":
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0, isolation_level=None)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")  # readers do not block the logger
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.executescript(SCHEMA)
    return conn


def _sql_value(v: float) -> float | None:
    # SQLite stores +-Inf as REAL but has no NaN; keep the row, drop the value.
    return None if math.isnan(v) else v


def record_success(conn: sqlite3.Connection, target: str, ts: float,
                   duration_s: float, exp: Exposition) -> int:
    meta = exp.sample_metadata()
    uptime = exp.value(UPTIME_METRIC)
    with conn:  # one transaction per scrape
        conn.execute("BEGIN")
        cur = conn.execute(
            "INSERT INTO scrapes(target, ts, ok, duration_s, error, uptime_s) "
            "VALUES (?, ?, 1, ?, NULL, ?)", (target, ts, duration_s, uptime))
        scrape_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO metrics(name, type, help) VALUES (?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET type = excluded.type, help = excluded.help "
            "WHERE type != excluded.type OR help != excluded.help",
            [(n, t, h) for n, (t, h) in meta.items()])
        # A repeated series in one exposition keeps the last value.
        conn.executemany(
            "INSERT OR REPLACE INTO samples(scrape_id, name, labels, value) VALUES (?, ?, ?, ?)",
            [(scrape_id, s.name, s.labels, _sql_value(s.value)) for s in exp.samples])
    return scrape_id


def record_failure(conn: sqlite3.Connection, target: str, ts: float,
                   duration_s: float, error: str) -> int:
    cur = conn.execute(
        "INSERT INTO scrapes(target, ts, ok, duration_s, error, uptime_s) "
        "VALUES (?, ?, 0, ?, ?, NULL)", (target, ts, duration_s, error))
    return cur.lastrowid


def prune(conn: sqlite3.Connection, retention_days: float, now: float) -> int:
    if retention_days <= 0:
        return 0
    with conn:
        conn.execute("BEGIN")
        cur = conn.execute("DELETE FROM scrapes WHERE ts < ?", (now - retention_days * 86400.0,))
    return cur.rowcount


# ---------------------------------------------------------------------------
# Scraping.

def metrics_url(target: str) -> str:
    """Accept either a server base URL or the full /metrics URL."""
    url = target.rstrip("/")
    if not re.match(r"^https?://", url):
        url = "http://" + url
    if not url.endswith("/metrics"):
        url += "/metrics"
    return url


def fetch(url: str, timeout: float) -> str:
    req = urllib.request.Request(url, headers={"Accept": "text/plain",
                                               "User-Agent": "ds4-metrics-logger"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace")


def scrape_once(conn: sqlite3.Connection, url: str, timeout: float) -> bool:
    ts = time.time()
    t0 = time.monotonic()
    try:
        text = fetch(url, timeout)
        exp = parse_exposition(text)
        if not exp.samples:
            raise ValueError("response contained no samples")
    except (urllib.error.URLError, OSError, ValueError) as e:
        duration = time.monotonic() - t0
        reason = getattr(e, "reason", None) or e
        record_failure(conn, url, ts, duration, str(reason))
        log.warning("scrape %s failed after %.2fs: %s", url, duration, reason)
        return False
    duration = time.monotonic() - t0
    record_success(conn, url, ts, duration, exp)
    log.debug("scrape %s: %d samples in %.3fs", url, len(exp.samples), duration)
    return True


def run(conn: sqlite3.Connection, urls: list[str], interval: float, timeout: float,
        retention_days: float, stop: threading.Event, max_scrapes: int = 0) -> None:
    next_prune = 0.0
    rounds = 0
    next_tick = time.monotonic()
    while not stop.is_set():
        for url in urls:
            if stop.is_set():
                break
            scrape_once(conn, url, timeout)
        rounds += 1
        now = time.time()
        if now >= next_prune:
            removed = prune(conn, retention_days, now)
            if removed:
                log.info("pruned %d scrapes older than %g days", removed, retention_days)
            next_prune = now + 3600.0
        if max_scrapes and rounds >= max_scrapes:
            break
        # Fixed cadence; if a round overran, skip the missed ticks instead of bursting.
        next_tick += interval
        now_mono = time.monotonic()
        if next_tick < now_mono:
            next_tick = now_mono + interval - ((now_mono - next_tick) % interval)
        stop.wait(next_tick - now_mono)


# ---------------------------------------------------------------------------
# CLI.

def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Poll ds4-server GET /metrics and log every sample to SQLite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    env_urls = _env("DS4_METRICS_URL", "http://127.0.0.1:8000")
    p.add_argument("--url", action="append", dest="urls", metavar="URL",
                   help="server base URL or full /metrics URL; repeat for several servers "
                        "(env DS4_METRICS_URL, comma-separated)")
    p.add_argument("--db", default=_env("DS4_METRICS_DB", "ds4-metrics.sqlite"),
                   help="SQLite database path (env DS4_METRICS_DB)")
    p.add_argument("--interval", type=float,
                   default=float(_env("DS4_METRICS_INTERVAL", "15")),
                   help="seconds between scrapes (env DS4_METRICS_INTERVAL)")
    p.add_argument("--timeout", type=float,
                   default=float(_env("DS4_METRICS_TIMEOUT", "10")),
                   help="HTTP timeout in seconds (env DS4_METRICS_TIMEOUT)")
    p.add_argument("--retention-days", type=float,
                   default=float(_env("DS4_METRICS_RETENTION_DAYS", "0")),
                   help="delete scrapes older than this; 0 keeps everything "
                        "(env DS4_METRICS_RETENTION_DAYS)")
    p.add_argument("--once", action="store_true",
                   help="scrape every target once and exit (non-zero if any scrape failed)")
    p.add_argument("--log-level", default=_env("DS4_METRICS_LOG_LEVEL", "INFO"),
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="(env DS4_METRICS_LOG_LEVEL)")
    p.set_defaults(env_urls=env_urls)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")
    raw = args.urls or [u for u in args.env_urls.split(",")]
    urls = list(dict.fromkeys(metrics_url(u.strip()) for u in raw if u.strip()))
    if not urls:
        log.error("no target URL given")
        return 2
    if args.interval <= 0:
        log.error("--interval must be positive")
        return 2

    conn = open_db(args.db)
    if args.once:
        ok = all([scrape_once(conn, u, args.timeout) for u in urls])
        conn.close()
        return 0 if ok else 1

    stop = threading.Event()

    def on_signal(signum, _frame):
        log.info("received %s, stopping", signal.Signals(signum).name)
        stop.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    log.info("logging %s every %gs to %s", ", ".join(urls), args.interval, args.db)
    try:
        run(conn, urls, args.interval, args.timeout, args.retention_days, stop)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
