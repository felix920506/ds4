#!/usr/bin/env python3
"""Tests for metrics_logger.py.  Run: python3 -m unittest -v test_metrics_logger"""

import http.server
import math
import os
import sqlite3
import tempfile
import threading
import unittest

import metrics_logger as ml

# Shape of ds4-server's GET /metrics (metrics_render_prometheus in ds4_server.c).
EXPOSITION = """\
# HELP ds4_decode_tokens_total Tokens committed by decode.
# TYPE ds4_decode_tokens_total counter
ds4_decode_tokens_total {decode}
# HELP ds4_decode_seconds_total Wall time in decode.
# TYPE ds4_decode_seconds_total counter
ds4_decode_seconds_total 12.500000
# HELP ds4_server_requests_total Accepted API requests by endpoint.
# TYPE ds4_server_requests_total counter
ds4_server_requests_total{{endpoint="chat"}} 3
ds4_server_requests_total{{endpoint="completions"}} 0
# HELP ds4_server_queue_wait_seconds Arrival to worker start.
# TYPE ds4_server_queue_wait_seconds histogram
ds4_server_queue_wait_seconds_bucket{{le="0.025"}} 1
ds4_server_queue_wait_seconds_bucket{{le="+Inf"}} 3
ds4_server_queue_wait_seconds_sum 0.420000
ds4_server_queue_wait_seconds_count 3
# HELP ds4_server_uptime_seconds Seconds since the server started.
# TYPE ds4_server_uptime_seconds gauge
ds4_server_uptime_seconds {uptime}
# HELP ds4_server_queued_requests Requests waiting for a slot.
# TYPE ds4_server_queued_requests gauge
ds4_server_queued_requests {queued}
"""


class FakeServer:
    def __init__(self):
        self.body = ""
        self.status = 200
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/metrics":
                    self.send_error(404)
                    return
                data = outer.body.encode()
                self.send_response(outer.status)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def set(self, decode, uptime, queued=0):
        self.body = EXPOSITION.format(decode=decode, uptime=uptime, queued=queued)

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class ParseTests(unittest.TestCase):
    def test_labels_types_and_specials(self):
        exp = ml.parse_exposition(
            '# TYPE h histogram\n'
            'h_bucket{le="+Inf",a="x\\"y"} 4\n'
            'h_sum 1.5\n'
            'h_count 4 1700000000000\n'
            'g NaN\n'
            'garbage line here\n')
        self.assertEqual([s.name for s in exp.samples], ["h_bucket", "h_sum", "h_count", "g"])
        self.assertEqual(exp.samples[0].labels, 'a="x\\"y",le="+Inf"')
        self.assertTrue(math.isnan(exp.samples[3].value))
        meta = exp.sample_metadata()
        self.assertEqual(meta["h_bucket"][0], "histogram")
        self.assertEqual(meta["h_count"][0], "histogram")
        self.assertEqual(meta["g"][0], "untyped")

    def test_metrics_url(self):
        self.assertEqual(ml.metrics_url("host:8000"), "http://host:8000/metrics")
        self.assertEqual(ml.metrics_url("http://h:1/"), "http://h:1/metrics")
        self.assertEqual(ml.metrics_url("https://h/metrics"), "https://h/metrics")


class LoggerTests(unittest.TestCase):
    def setUp(self):
        self.srv = FakeServer()
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "sub", "m.sqlite")
        self.conn = ml.open_db(self.db_path)
        self.url = ml.metrics_url(self.srv.url)

    def tearDown(self):
        self.conn.close()
        self.srv.close()
        self.tmp.cleanup()

    def scrape(self, ts, **kw):
        self.srv.set(**kw)
        # Pin scrape time so rates are deterministic.
        real = ml.time.time
        ml.time.time = lambda: ts
        try:
            return ml.scrape_once(self.conn, self.url, timeout=5)
        finally:
            ml.time.time = real

    def deltas(self, name, labels=""):
        return self.conn.execute(
            "SELECT ts, value, delta, rate FROM sample_deltas "
            "WHERE name = ? AND labels = ? ORDER BY ts", (name, labels)).fetchall()

    def test_scrape_stores_samples_and_metadata(self):
        self.assertTrue(self.scrape(1000.0, decode=100, uptime=50))
        row = self.conn.execute("SELECT ok, uptime_s, target FROM scrapes").fetchone()
        self.assertEqual(row, (1, 50.0, self.url))
        n = self.conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        self.assertEqual(n, 10)
        self.assertEqual(
            self.conn.execute("SELECT type, help FROM metrics WHERE name = "
                              "'ds4_server_queue_wait_seconds_bucket'").fetchone(),
            ("histogram", "Arrival to worker start."))
        self.assertEqual(
            self.conn.execute("SELECT value FROM samples WHERE name = "
                              "'ds4_server_requests_total' AND labels = 'endpoint=\"chat\"'")
            .fetchone()[0], 3.0)

    def test_rates_and_restart(self):
        self.scrape(1000.0, decode=100, uptime=50, queued=2)
        self.scrape(1010.0, decode=300, uptime=60, queued=5)
        # Restart: uptime drops; the counter is already above its old value.
        self.scrape(1020.0, decode=400, uptime=5, queued=1)
        self.scrape(1030.0, decode=450, uptime=15, queued=0)
        rows = self.deltas("ds4_decode_tokens_total")
        self.assertEqual(rows[0][2:], (None, None))
        self.assertEqual(rows[1][2:], (200.0, 20.0))
        self.assertEqual(rows[2][2:], (None, None))
        self.assertEqual(rows[3][2:], (50.0, 5.0))
        gauge = self.deltas("ds4_server_queued_requests")
        self.assertEqual([r[2] for r in gauge], [None, 3.0, -4.0, -1.0])
        self.assertTrue(all(r[3] is None for r in gauge))

    def test_failure_is_recorded(self):
        self.srv.close()
        self.assertFalse(self.scrape(1000.0, decode=1, uptime=1))
        ok, err = self.conn.execute("SELECT ok, error FROM scrapes").fetchone()
        self.assertEqual(ok, 0)
        self.assertTrue(err)
        self.srv = FakeServer()  # for tearDown

    def test_prune_cascades(self):
        self.scrape(1000.0, decode=1, uptime=1)
        self.scrape(200000.0, decode=2, uptime=2)
        self.assertEqual(ml.prune(self.conn, 1.0, now=200000.0), 1)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM scrapes").fetchone()[0], 1)
        self.assertEqual(self.conn.execute(
            "SELECT COUNT(DISTINCT scrape_id) FROM samples").fetchone()[0], 1)

    def test_main_once_and_run_loop(self):
        self.srv.set(decode=7, uptime=3)
        db = os.path.join(self.tmp.name, "cli.sqlite")
        self.assertEqual(ml.main(["--url", self.srv.url, "--db", db, "--once",
                                  "--log-level", "ERROR"]), 0)
        conn = ml.open_db(db)
        stop = threading.Event()
        ml.run(conn, [self.url], interval=0.01, timeout=5, retention_days=0,
               stop=stop, max_scrapes=3)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM scrapes WHERE ok").fetchone()[0], 4)
        conn.close()
        # Readers see the data while the WAL is in use.
        ro = sqlite3.connect(db)
        self.assertEqual(ro.execute("SELECT COUNT(*) FROM samples").fetchone()[0], 40)
        ro.close()


if __name__ == "__main__":
    unittest.main()
