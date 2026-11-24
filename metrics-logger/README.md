# ds4 metrics logger

Polls `GET /metrics` on one or more `ds4-server` instances and appends every
sample to a SQLite database. Use it to keep a performance history without
running Prometheus. It uses only the Python standard library (3.10+).

```sh
python3 metrics_logger.py --url http://127.0.0.1:8000 --db ds4-metrics.sqlite --interval 15
```

| Flag | Env var | Default |
| --- | --- | --- |
| `--url URL` (repeatable) | `DS4_METRICS_URL` (comma-separated) | `http://127.0.0.1:8000` |
| `--db PATH` | `DS4_METRICS_DB` | `ds4-metrics.sqlite` |
| `--interval S` | `DS4_METRICS_INTERVAL` | `15` |
| `--timeout S` | `DS4_METRICS_TIMEOUT` | `10` |
| `--retention-days D` (0 = keep all) | `DS4_METRICS_RETENTION_DAYS` | `0` |
| `--log-level` | `DS4_METRICS_LOG_LEVEL` | `INFO` |
| `--once` | | scrape once and exit |

A URL can be the server base or the full `/metrics` URL. A failed scrape is
stored as a row with `ok = 0` and the error message, so gaps are visible.

## Container

CI publishes `ghcr.io/<owner>/ds4-metrics-logger` (`latest` and `sha-<short>`
from `main`, `<version>` from `v*` tags) for `linux/amd64` and `linux/arm64`.

```sh
docker run -d --name ds4-metrics --restart unless-stopped --network host \
  -v ds4-metrics:/data \
  -e DS4_METRICS_URL=http://127.0.0.1:8000 \
  ghcr.io/<owner>/ds4-metrics-logger:latest
```

The database is `/data/ds4-metrics.sqlite`. It is in WAL mode, so you can
query it while the logger runs:
`docker run --rm -v ds4-metrics:/data -it keinos/sqlite3 sqlite3 /data/ds4-metrics.sqlite`.

## Schema

- `scrapes(id, target, ts, ok, duration_s, error, uptime_s)`: one row per poll
  (`ts` is unix seconds).
- `metrics(name, type, help)`: Prometheus metadata per sample name.
- `samples(scrape_id, name, labels, value)`: `labels` is the sorted label
  set, e.g. `endpoint="chat"`, or `''`.
- `sample_deltas`: a view that pairs each sample with the previous scrape of
  the same series and adds `interval_s`, `delta` and `rate` (per second,
  counters and histograms only). Counter deltas are `NULL` across a server
  restart, which is detected from `ds4_server_uptime_seconds`.

Counter meanings are documented in [docs/SERVER.md](../docs/SERVER.md#metrics).
Engine counters (`ds4_prefill_*`, `ds4_decode_*`) move only when a prefill
or decode call returns. A long prefill therefore shows up as a single jump
in one interval, not as a steady rate. For throughput, divide token deltas
by the matching `*_seconds_total` deltas, as in the queries below.

```sql
-- decode and prefill throughput per interval (tokens / second of wall time)
SELECT datetime(ts, 'unixepoch') AS t, name, round(rate, 1)
  FROM sample_deltas
 WHERE name IN ('ds4_decode_tokens_total', 'ds4_prefill_tokens_total')
 ORDER BY ts;

-- decode speed while decoding: tokens per decode-second, per interval
SELECT datetime(t.ts, 'unixepoch'), t.delta / s.delta AS decode_tok_s
  FROM sample_deltas t JOIN sample_deltas s USING (scrape_id)
 WHERE t.name = 'ds4_decode_tokens_total' AND s.name = 'ds4_decode_seconds_total'
   AND s.delta > 0;

-- mean time to first token per interval
SELECT datetime(a.ts, 'unixepoch'), a.delta / c.delta AS ttft_s
  FROM sample_deltas a JOIN sample_deltas c USING (scrape_id)
 WHERE a.name = 'ds4_server_time_to_first_token_seconds_sum'
   AND c.name = 'ds4_server_time_to_first_token_seconds_count' AND c.delta > 0;
```

## Tests

```sh
python3 -m unittest -v test_metrics_logger
```
