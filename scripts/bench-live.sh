#!/usr/bin/env bash
# bench-live.sh — end-to-end latency against a RUNNING gateway.
#
# Port of rust-old's `benches/e2e.rs` + `benches/live_providers.rs`. Like those,
# this is OPT-IN and not part of CI: it needs a live hoosh and a live backend,
# so it measures a machine and a model, not the code — putting it in the release
# gate would make the gate depend on whatever happens to be running.
# `tests/hoosh.bcyr` stays the pure-logic suite that CI enforces.
#
# What CI's benches cannot tell you: the pure-logic numbers are nanoseconds,
# while a real request is milliseconds dominated by the provider. This answers
# "how much of that is hoosh" — the gateway overhead between accept and forward.
#
# READ THE NUMBERS WITH THIS IN MIND: each iteration forks a `curl`, and process
# startup is ~5 ms on a typical Linux box. So a floor of roughly 5 ms/req here
# is the HARNESS, not the gateway — against a local mock backend every row lands
# at ~5.4 ms. Use this for RELATIVE comparisons (cached vs uncached, before vs
# after a change, one provider vs another), not as an absolute latency figure.
# For absolute per-request cost, read `hoosh bench <url>` (raw connect latency)
# or the nanosecond suite in tests/hoosh.bcyr.
#
# Usage:
#   scripts/bench-live.sh [url] [model] [iterations]
#   scripts/bench-live.sh http://127.0.0.1:8088 llama3 20
set -u

URL="${1:-http://127.0.0.1:8088}"
MODEL="${2:-llama3}"
N="${3:-20}"

command -v curl >/dev/null 2>&1 || { echo "bench-live: curl not found"; exit 1; }

if ! curl -sf -o /dev/null --max-time 5 "$URL/v1/health"; then
    echo "bench-live: no hoosh at $URL (start one with: hoosh serve)"
    exit 1
fi

echo "=== hoosh live benchmark ==="
echo "  gateway:    $URL"
echo "  model:      $MODEL"
echo "  iterations: $N"
echo ""

# --- Control-plane endpoints: pure gateway work, no provider involved. -------
# These are the honest measure of hoosh's own overhead.
bench_get() {
    local path="$1" label="$2" iters="$3"
    local start end total
    start=$(date +%s%N)
    local k=0
    while [ "$k" -lt "$iters" ]; do
        curl -sf -o /dev/null --max-time 10 "$URL$path" || { echo "  $label: FAILED"; return 1; }
        k=$((k + 1))
    done
    end=$(date +%s%N)
    total=$(( (end - start) / 1000000 ))
    printf "  %-26s %6s ms total  %8s us/req\n" "$label" "$total" "$(( (end - start) / 1000 / iters ))"
}

echo "-- control plane (no provider) --"
bench_get /v1/health          "health"            "$N"
bench_get /v1/models          "models"            "$N"
bench_get /v1/cache/stats     "cache_stats"       "$N"
bench_get /v1/costs           "costs"             "$N"
bench_get /v1/health/providers "health_providers" "$N"

# --- Inference: gateway + provider. -----------------------------------------
echo ""
echo "-- inference (gateway + provider) --"
BODY=$(printf '{"model":"%s","messages":[{"role":"user","content":"Say OK."}],"max_tokens":16}' "$MODEL")

start=$(date +%s%N)
ok=0
fail=0
i=0
while [ "$i" -lt "$N" ]; do
    if curl -sf -o /dev/null --max-time 120 -X POST "$URL/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$BODY"; then
        ok=$((ok + 1))
    else
        fail=$((fail + 1))
    fi
    i=$((i + 1))
done
end=$(date +%s%N)

if [ "$ok" -eq 0 ]; then
    echo "  chat_completions: all $N requests FAILED — is a backend reachable?"
    exit 1
fi

total_ms=$(( (end - start) / 1000000 ))
printf "  %-26s %6s ms total  %8s ms/req  (%s ok, %s failed)\n" \
    "chat_completions" "$total_ms" "$(( total_ms / ok ))" "$ok" "$fail"

# Cache-warm repeat: same body twice. With [cache] enabled the second pass is
# served by hoosh alone, so the delta is the provider's share of the latency.
echo ""
echo "-- cached repeat (same prompt) --"
curl -sf -o /dev/null --max-time 120 -X POST "$URL/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$BODY" >/dev/null 2>&1
start=$(date +%s%N)
i=0
while [ "$i" -lt "$N" ]; do
    curl -sf -o /dev/null --max-time 120 -X POST "$URL/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$BODY" >/dev/null 2>&1
    i=$((i + 1))
done
end=$(date +%s%N)
cached_ms=$(( (end - start) / 1000000 ))
printf "  %-26s %6s ms total  %8s ms/req\n" "chat_cached" "$cached_ms" "$(( cached_ms / N ))"
echo ""
echo "  (a large gap between chat_completions and chat_cached is the provider's"
echo "   share; a small one means the cache is off or the key is not matching)"
echo ""
echo "done"
