# Serving Models

[README](../README.md) | [Client setup](CLIENTS.md)

## Start locally

```sh
./ds4-server --ctx 32768
```

The default address is `http://127.0.0.1:8000`. Use `--host 0.0.0.0` to listen
on other interfaces. Restrict access to trusted clients; for an Internet-facing
deployment, put authentication and TLS in front of the server.

`--cors` enables browser cross-origin headers. It does not change the listening
address or provide access control.

The selected build chooses the backend. Pass `-m FILE` for an explicit model.
Use `--chdir /path/to/ds4` when starting outside the project directory so
relative runtime files such as Metal kernels can be found.

## APIs

| Endpoint | Use |
| --- | --- |
| `GET /v1/models` | Loaded model information |
| `POST /v1/chat/completions` | OpenAI-style chat |
| `POST /v1/responses` | Responses-style requests and continuations |
| `POST /v1/completions` | Text completions |
| `POST /v1/messages` | Anthropic-style messages |

The Flash and PRO names accepted by the model endpoints are compatibility
aliases, not separate loaded models. The GGUF passed at startup selects the model.

`GET /v1/models` and `GET /v1/models/{id}` include `architecture.input_modalities` and `architecture.output_modalities`. Input modalities are `["text", "image"]` when the loaded engine has vision enabled, otherwise `["text"]`; output modalities are always `["text"]`. Compatibility aliases report the same loaded-engine capabilities.

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Explain Redis streams."}],"stream":true}'
```

Chat, Responses, and Anthropic support tools and SSE streaming. Reasoning is
returned separately from visible text in each API's native form. Standard
sampling and output-budget fields are supported; explicit request parameters
take precedence over defaults.

The default sampling settings are temperature 1, top-p 1, and min-p 0.05.
For DeepSeek, thinking is on by default. `reasoning_effort=max` selects Think
Max only with sufficient context; otherwise it falls back to normal thinking.
`xhigh` maps to normal thinking, not Think Max. Use `think:false`, a disabled
thinking object, or a non-thinking model alias for direct answers.

## Multiple sessions

```sh
./ds4-server --ctx 4096 --batched-session 4
```

Without `--batched-session`, there is one resident session. With it, the server
has that many slots, each with an independent KV state, and queues requests
when all slots are busy. One session is created at startup, which checks that
the context fits; the other slots get theirs on their first request.

A context that fits once may not fit four times. When an active request needs
memory that is not there, the server kicks idle slots: it writes the slot's
live state to the disk KV cache and frees its session. The conversation is not
lost; its next request restores it from disk on whichever slot is free,
through the same lookup as after a restart. Kicks take the cheapest idle slot
first: one with no conversation, then one whose state is already on disk, then
one that has to be written, the longest idle first. A slot that runs a request,
or that a queued request can only continue on, is never kicked. If no idle slot
is left, a request that needs a new session waits for a running request to
finish and takes that slot's memory then (`ds4_server_memory_waits_total`).
It fails with HTTP 503 only when no other slot could give memory back.

Kicking needs `--kv-disk-dir`. Without it, and for state the disk cache cannot
hold (conversations shorter than `--kv-cache-min-tokens`), a
kicked conversation is prefilled again; such slots are kicked last, and the log
says so. A client that continues a tool call by id alone, without replaying the
history, gets the usual 409 after a kick, as it does after any slot reuse.
Distributed and tensor-parallel servers create all sessions at startup and do
not kick.

Where the model supports it, the slots share one prefill workspace instead of
each keeping its own, so an extra slot costs only its caches. That matters most
for Qwen3.8 Flash Next, whose transients are sized by the prefill chunk rather
than by the context: at the default chunk they run to several GiB per session.
The startup line reports both figures.

| Backend/model | Decode execution |
| --- | --- |
| Metal, resident Flash | Native shared-expert/QKV batching where supported |
| Metal, resident V4.1 Flash | Native decoding for 2-8 sessions |
| Metal RDMA TP, V4.1 Flash | Native decoding for 3-8 sessions; ordered fallback for two |
| Metal SSD streaming, V4.1 Flash | Ordered fallback |
| Metal, GLM 5.2 | Ordered fallback |
| Metal, GLM 5.3 | Native batching through 2051 visible tokens; ordered fallback afterward |
| Metal, Qwen3.8 Flash Next with resident weights | Native batching of the shared work; recurrent state, caches and PLE history stay per session |
| CUDA, supported multi-GPU Flash TP layout | Native grouped decode and mixed prefill/decode |
| Metal, Qwen3.8 Flash Next with SSD expert streaming | Ordered fallback with per-session streaming workspaces |
| Single-GPU CUDA, including Spark | Ordered fallback |

Fallback executes the rows separately. It provides concurrency and scheduling
fairness, not the aggregate speedup of native batching. Native grouping may
change floating-point reduction order slightly. V4.1 sessions containing images
use the ordered fallback.

Long prefills yield to active decoders in bounded intervals, normally 128
tokens. `--mixed-prefill-quantum N` changes that interval for testing. While
nothing is decoding, a V4.1 slot prefills in the engine's own chunk.
Session-batched serving uses ordinary target decoding, except Qwen3.8 on
Metal, where `--mtp` also batches speculative decoding with resident weights
and uses ordered speculative cycles with SSD expert streaming. Its
`--mtp-exact-sampling` mode uses ordinary batches for nonzero-temperature
requests. Other models do not use MTP/DSpark while session batching is active.
For the eight-L40S example, see [CUDA GPUs](CUDA_MULTI_GPU.md#serve-multiple-users).

## Images

Start with the matching language GGUF and `--vision FILE`; see
[model-specific instructions](MODELS.md#vision).

OpenAI chat and Responses accept inline PNG, JPEG, WebP, GIF, TIFF and BMP
data URIs. Anthropic accepts base64 image sources of the same types. Formats
and variants the built-in PNG/JPEG decoder cannot read (16-bit or interlaced
PNG, CMYK JPEG, and every type other than PNG and JPEG) are decoded by
ImageMagick when it is installed; otherwise the request fails with an error
naming the reason. See `DS4_IMAGE_DECODER` in the [README](../README.md). Remote URLs and server-side file paths are rejected.
Image blocks preserve their order in the request. A request is rejected when
its tokens, image blocks included, do not fit the context window; the only
other limit is a ceiling of 4096 images per request, far above what any
context window holds, so that a malformed request cannot reserve memory for
images it never sends. The HTTP body is limited to 4 GiB, so long sessions
that replay their full screenshot history each turn fit. The body buffer grows
as bytes arrive, so a client that advertises a large Content-Length without
sending the body does not reserve that memory.

## Disk KV cache

Disk caching saves useful prefixes across slot reuse and server restarts:

```sh
./ds4-server --ctx 100000 \
  --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

Clients can resend complete conversation histories. The server first tries
the live token prefix, then compatible rendered-text prefixes from disk, and
prefills the new suffix. With multiple slots, each slot has its own live state;
disk is the additional persistence layer, not the only way to retain sessions.
It is also where a slot's conversation goes when the slot is kicked for memory
(see Multiple sessions), so size `--kv-disk-space-mb` for the conversations
that should survive that.

When the budget is full, the least recently used checkpoints go first: one
that was stored or loaded a minute ago outlives one idle for ten, and one from
today outlives one from last month, whatever its size or old hit count. Three things adjust that order. Recent hits and a
checkpoint's tokens per byte count in its favour; a cold checkpoint, which new
conversations with the same system prompt start from, gets a bonus; and a
checkpoint that a longer one of the same conversation already extends is
treated as mostly redundant, so a conversation gives up its own old waypoints
before it costs another conversation its only one. A client that rewrites
the model's replies leaves checkpoints that end in text it will never send
again; once the cache has seen one go unused, it keeps what that conversation
can still resume from instead. Each line
`kv cache evicted reason=disk-cache-full` in the log names what was removed.
`make kv-evict-sim` (`make kv-evict-sim-rocm` on a ROCm build) replays
simulated workloads against the eviction code and reports how many tokens
would have to be prefilled again; use it when changing the policy.

Conversations with images are cached like any other. A checkpoint records the
fingerprint of every image it was built from, and a request restores it only
when its own images are the same ones at the same places: the same prompt with
a different image is a miss, not a wrong answer. Tensor-parallel servers do not
cache image conversations, because their cache loading rebuilds from tokens.

TP cache loading rebuilds the saved token prefix on both ranks. It is not an
instantaneous restoration of both GPUs. Pipeline loading redistributes the
saved layer state over its route. A pipeline server saves checkpoints only
when every stage runs with the same `--ctx` and `--prefill-chunk`; see
[Pipeline parallelism](DISTRIBUTED.md#pipeline-parallelism).

Defaults are intended to avoid saving fragile token boundaries. For unusual
workloads, the controls are `--kv-cache-min-tokens`,
`--kv-cache-cold-max-tokens`, `--kv-cache-continued-interval-tokens`,
`--kv-cache-boundary-trim-tokens`, and `--kv-cache-boundary-align-tokens`.
Check `./ds4-server --help` for their defaults.

During a long prefill or generation a continued checkpoint is due every
`--kv-cache-continued-interval-tokens` tokens, rounded up to a multiple of the
boundary alignment; the startup line reports the interval in effect as
`continued=N`. A checkpoint can only be written while the session stands
exactly on such a token count. With `--batched-session` a prefill stops at
each of them. Without it, one is written only where a prefill chunk happens to
end there.

Quantization variants may share compatible prefixes. Add
`--kv-cache-reject-different-quant` for same-quant reuse only.
Cache files contain prompt text and model state: treat the directory as
private. It is disposable; stop the server before clearing it.

## Metrics

`GET /metrics` returns Prometheus text and `GET /metrics/json` returns the
same counters as JSON with derived rates. Both are cumulative since startup
and cheap to scrape. The engine counts wall time per phase at its public API,
so a request's cost is split into:

| Counter family | Meaning |
| --- | --- |
| `ds4_prefill_*` | Prompt syncs: tokens evaluated, tokens reused from the live KV prefix, seconds |
| `ds4_decode_*` | Forward passes, session rows, committed tokens, seconds; tokens per pass is the speculative gain and rows per pass the batch size |
| `ds4_vision_*` | Images encoded and seconds |
| `ds4_kv_save_*`, `ds4_kv_load_*` | Disk KV payloads written and restored, bytes, seconds |
| `ds4_net_*` | Bytes and blocked seconds exchanged with other nodes (pipeline or TP), gate count, and worker compute reported in pipeline telemetry |
| `ds4_ssd_read_*` | Routed-expert bytes streamed from disk and seconds |
| `ds4_server_*` | Requests by endpoint and finish reason, prompt/cached/completion tokens, queue-wait, time-to-first-token and request-duration histograms, disk KV and image cache hits, slot kicks (`ds4_server_slot_kicks_total`, `state="saved"` when nothing was lost and `"dropped"` when the conversation must be prefilled again; `ds4_server_slot_kick_bytes_total`), live gauges including `ds4_server_resident_sessions` |

On a distributed coordinator the network wait is nested inside prefill and
decode time; `network_transport_seconds_estimate` in the JSON view subtracts
the compute the workers reported, leaving transport latency. The JSON
`time_share` object gives the fraction of uptime spent in each phase, which
is the number to look at when deciding whether a deployment is bound by
prompt processing, generation, or idle waiting for clients.

```sh
curl -s http://127.0.0.1:8000/metrics | grep -E '^ds4_(prefill|decode)_'
curl -s http://127.0.0.1:8000/metrics/json | python3 -m json.tool
```

A kicked conversation that returns is an ordinary disk KV hit
(`ds4_server_kv_disk_lookups_total{result="hit"}`, `ds4_kv_load_*`); there is
no separate restore counter.

`--metrics-log 60` additionally logs one line per minute with the interval's
deltas: prefill and decode throughput, time shares, network and disk I/O,
requests completed, and average time to first token and queue wait. The
interactive `ds4` CLI exposes the same engine accounting with `/stats`.

## Tool history and debugging

For DeepSeek, the server preserves sampled DSML tool blocks and assigns
unguessable tool IDs. Replaying those IDs avoids retokenizing a differently
formatted JSON history. The bounded replay map can be stored in cache files.
When exact replay is unavailable, canonical rendering may require rebuilding
part of the prefix.

`--tool-memory-max-ids` bounds this map.
`--disable-exact-dsml-tool-replay` disables it for diagnostic comparisons.
Use `--trace /tmp/ds4-trace.txt` to record prompt rendering, cache decisions,
generated text, and tool-parser events. Traces can contain sensitive content.

Cache formats are implementation details. The current header and extension
definitions are in [ds4_kvstore.h](../ds4_kvstore.h) and
[ds4_kvstore.c](../ds4_kvstore.c); model-specific payload handling is in
[ds4.c](../ds4.c).
