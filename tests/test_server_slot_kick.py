#!/usr/bin/env python3
"""Live check that a kicked slot comes back from the disk KV cache.

Needs a server started with several slots, a disk KV cache and the diagnostic
memory-pressure endpoint:

    DS4_SERVER_DEBUG_MEMORY_PRESSURE=1 ./ds4-server --batched-session 3 \\
        --ctx 32768 --kv-disk-dir /tmp/ds4-slotkick/kv ...

It holds one greedy conversation per slot, optionally kicks slots between the
first and the second turn, and then continues every conversation. Every
continuation must report cached prompt tokens: a kicked conversation is
restored from disk, not prefilled again. With --kick the server's memory must
drop after the kick. On a server whose context is too large for all slots to
hold a session at once, the kicks happen on their own; --min-kicks checks
that they did.

With --concurrent the conversations of a turn are sent at once. On a server
that cannot hold a session per slot (DS4_SERVER_DEBUG_MAX_SESSIONS=1 stands in
for that), a request then finds every other slot busy and has to wait for one
to finish; --min-waits checks that it did, and every request must still
succeed.

Run it without --kick on a fresh server and KV directory with --out, then
with --kick and --expect on another fresh server: the second-turn text must
match. Start both servers with --prefill-chunk 16; larger ROCm prefill chunks
are not bit-reproducible from one process to the next, kick or no kick.
"""

import argparse
import glob
import json
import sys
import threading
import urllib.request

TOPICS = ("tide pools", "steam locomotives", "alpine glaciers", "bronze casting")


def http(base, path, body=None, timeout=900):
    data = None if body is None else (
        body if isinstance(body, bytes) else json.dumps(body).encode())
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def first_message(index, words):
    topic = TOPICS[index % len(TOPICS)]
    filler = " ".join("%s fact %d." % (topic, i) for i in range(words // 3))
    return ("Conversation %d is about %s. Keep these notes in mind:\n%s\n"
            "Write two sentences about %s." % (index, topic, filler, topic))


def chat(base, messages, max_tokens):
    out = http(base, "/v1/chat/completions", {
        "model": "deepseek-chat",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 7,
    })
    usage = out.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return {
        "text": out["choices"][0]["message"].get("content") or "",
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "cached_tokens": details.get("cached_tokens", 0),
    }


def chat_all(base, conversations, max_tokens, concurrent):
    """One chat per conversation, in order or all at once; results in order."""
    results = [None] * len(conversations)

    def run(i):
        try:
            results[i] = chat(base, conversations[i], max_tokens)
        except Exception as exc:  # reported by the caller, per conversation
            results[i] = exc

    if concurrent:
        threads = [threading.Thread(target=run, args=(i,))
                   for i in range(len(conversations))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        for i in range(len(conversations)):
            run(i)
    return results


def memory(pid):
    """GPU buffer bytes and anonymous RSS of the server, when readable."""
    out = {}
    if not pid:
        return out
    try:
        with open("/proc/%d/status" % pid) as fp:
            for line in fp:
                if line.startswith("RssAnon:"):
                    out["rss_anon"] = int(line.split()[1]) * 1024
    except OSError:
        pass
    # In a container the kfd entry carries the host pid; with one GPU
    # process running it is the only entry.
    procs = glob.glob("/sys/class/kfd/kfd/proc/%d" % pid) or \
        glob.glob("/sys/class/kfd/kfd/proc/[0-9]*")
    total = 0
    found = False
    for path in glob.glob(procs[0] + "/vram_*") if len(procs) == 1 else []:
        try:
            with open(path) as fp:
                total += int(fp.read().strip())
                found = True
        except (OSError, ValueError):
            pass
    if found:
        out["vram"] = total
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--conversations", type=int, default=3)
    ap.add_argument("--words", type=int, default=900,
                    help="filler words in the first message")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--kick", type=int, default=0,
                    help="slots to kick between the two turns")
    ap.add_argument("--restore-locks", action="store_true",
                    help="kick the way the engine asks during a disk restore")
    ap.add_argument("--min-kicks", type=int, default=0,
                    help="fail unless the server kicked at least this often")
    ap.add_argument("--concurrent", action="store_true",
                    help="send the conversations of a turn at once")
    ap.add_argument("--min-waits", type=int, default=0,
                    help="fail unless this many requests waited for memory")
    ap.add_argument("--pid", type=int, default=0, help="server pid, for memory")
    ap.add_argument("--out", help="write the second-turn results here")
    ap.add_argument("--expect", help="second-turn results of an un-kicked run")
    args = ap.parse_args()
    failures = []

    start = http(args.base, "/metrics/json")["server"]
    histories = [[{"role": "user", "content": first_message(i, args.words)}]
                 for i in range(args.conversations)]
    first = chat_all(args.base, histories, args.max_tokens, args.concurrent)
    for i, (messages, r) in enumerate(zip(histories, first)):
        if isinstance(r, Exception):
            print("FAIL: conversation %d first turn: %s" % (i, r))
            print("slot kick test: FAILED")
            return 1
        print("turn 1 conv %d: prompt=%d cached=%d text=%r" %
              (i, r["prompt_tokens"], r["cached_tokens"], r["text"][:60]))
        messages.append({"role": "assistant", "content": r["text"]})
        messages.append({"role": "user",
                         "content": "Now add one more sentence, then stop."})

    before = http(args.base, "/metrics/json")["server"]
    mem_before = memory(args.pid)
    print("before kick: resident_sessions=%d memory=%s" %
          (before["resident_sessions"], mem_before))

    if args.kick:
        for _ in range(args.kick):
            freed = http(args.base, "/debug/memory-pressure",
                         b"1 restore" if args.restore_locks else b"1")["freed_bytes"]
            print("kick: freed_bytes=%d" % freed)
            if freed == 0:
                failures.append("a kick released nothing")
        after = http(args.base, "/metrics/json")["server"]
        mem_after = memory(args.pid)
        kicks = after["slot_kicks"]
        kicked = before["resident_sessions"] - after["resident_sessions"]
        print("after kick: resident_sessions=%d slot_kicks=%s memory=%s" %
              (after["resident_sessions"], kicks, mem_after))
        if kicked != args.kick:
            failures.append("resident sessions fell by %d, wanted %d" %
                            (kicked, args.kick))
        if kicks["saved"] - before["slot_kicks"]["saved"] != args.kick:
            failures.append("kicks were not all saved: %s" % kicks)
        for key in mem_before:
            if key in mem_after:
                print("  %s: %.2f -> %.2f GiB" % (
                    key, mem_before[key] / 2**30, mem_after[key] / 2**30))
        if "vram" in mem_before and "vram" in mem_after and \
                mem_after["vram"] >= mem_before["vram"]:
            failures.append("GPU buffer bytes did not drop after the kick")

    results = chat_all(args.base, histories, args.max_tokens, args.concurrent)
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print("FAIL: conversation %d second turn: %s" % (i, r))
            print("slot kick test: FAILED")
            return 1
        print("turn 2 conv %d: prompt=%d cached=%d text=%r" %
              (i, r["prompt_tokens"], r["cached_tokens"], r["text"][:60]))
        if r["cached_tokens"] <= 0:
            failures.append("conversation %d was prefilled from scratch" % i)
    final = http(args.base, "/metrics/json")["server"]
    restores = final["kv_disk"]["hits"] - start["kv_disk"]["hits"]
    kicks = sum(final["slot_kicks"][k] - start["slot_kicks"][k]
                for k in ("saved", "dropped"))
    print("total: kicks=%d disk restores=%d resident_sessions=%d" %
          (kicks, restores, final["resident_sessions"]))
    # Every conversation was continued after the kicks, so each kicked one
    # came back from disk unless it is the one left without a session now.
    on_disk_only = args.conversations - final["resident_sessions"]
    if restores != kicks - on_disk_only:
        failures.append("%d kicks, %d conversations without a session, but "
                        "%d disk restores" % (kicks, on_disk_only, restores))
    waits = final["memory_waits"]["count"] - start["memory_waits"]["count"]
    print("memory waits: %d (%.1f s)" % (
        waits, final["memory_waits"]["seconds"] - start["memory_waits"]["seconds"]))
    if waits < args.min_waits:
        failures.append("only %d requests waited for memory, wanted at least %d"
                        % (waits, args.min_waits))
    if kicks < args.min_kicks:
        failures.append("only %d kicks, wanted at least %d" %
                        (kicks, args.min_kicks))

    if args.out:
        with open(args.out, "w") as fp:
            json.dump(results, fp, indent=1)
    if args.expect:
        with open(args.expect) as fp:
            expected = json.load(fp)
        for i, (got, want) in enumerate(zip(results, expected)):
            if got["text"] != want["text"]:
                failures.append("conversation %d second turn differs from the "
                                "un-kicked run:\n  got  %r\n  want %r" %
                                (i, got["text"], want["text"]))
            if got["prompt_tokens"] != want["prompt_tokens"]:
                failures.append("conversation %d prompt tokens %d != %d" % (
                    i, got["prompt_tokens"], want["prompt_tokens"]))

    for failure in failures:
        print("FAIL: " + failure)
    print("slot kick test: %s" % ("FAILED" if failures else "ok"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
