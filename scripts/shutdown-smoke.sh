#!/usr/bin/env bash
# shutdown-smoke.sh — SIGTERM stops a running gateway, cleanly, in the cases that have broken.
#
# main.cyr cannot be linked into tests/hoosh.tcyr, so its shutdown path is checked here against a real process:
#   - plain:    serve, SIGTERM.
#   - reload:   serve with `-p` on a port the config does not name, POST /v1/admin/reload, SIGTERM. Through 2.9.12
#               the reload rewrote _http_port, SIGTERM's wake-up connect went to the config's port, accept() never
#               woke, and the gateway ignored every later SIGTERM: only SIGKILL stopped it.
#   - any-addr: bound to 0.0.0.0, SIGTERM (the wake-up must reach a wildcard listener too).
# Each case must exit by itself within the limit and log "shutdown: complete". `timeout -k` stands behind every
# process, so a regression fails this script instead of leaving a gateway running.
#
# Usage: scripts/shutdown-smoke.sh [path/to/hoosh]   (default build/hoosh)
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
BIN="${1:-build/hoosh}"
[ -x "$BIN" ] || { echo "shutdown-smoke: no binary at $BIN"; exit 1; }
LIMIT_S=8

T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
cat > "$T/hoosh.cyml" <<'EOF'
[server]
bind = "127.0.0.1"
port = 8088
health_check_interval_secs = 0
EOF

fail=0

# run_case NAME PORT RELOAD(0|1) BIND
run_case() {
    local name=$1 port=$2 reload=$3 bind=$4
    local log="$T/$name.log"
    timeout -k 2 $((LIMIT_S + 20)) "$BIN" serve -p "$port" --bind "$bind" -c "$T/hoosh.cyml" >"$log" 2>&1 &
    local tpid=$!
    local up=0
    for _ in $(seq 1 50); do
        if curl -s -o /dev/null "http://127.0.0.1:$port/"; then up=1; break; fi
        sleep 0.1
    done
    if [ $up -eq 0 ]; then
        echo "FAIL $name: the gateway never answered on port $port"; cat "$log"; fail=1; wait "$tpid"; return
    fi
    if [ "$reload" = "1" ]; then
        curl -s -X POST "http://127.0.0.1:$port/v1/admin/reload" >/dev/null
    fi
    local gpid
    gpid=$(pgrep -P "$tpid" | head -1)
    kill -TERM "$gpid"
    local waited=0
    while kill -0 "$gpid" 2>/dev/null && [ $waited -lt $((LIMIT_S * 10)) ]; do
        sleep 0.1
        waited=$((waited + 1))
    done
    if kill -0 "$gpid" 2>/dev/null; then
        echo "FAIL $name: still running ${LIMIT_S}s after SIGTERM"; fail=1
    elif ! grep -q "shutdown: complete" "$log"; then
        echo "FAIL $name: exited without a clean shutdown"; cat "$log"; fail=1
    else
        echo "  ok $name: exited $(( waited * 100 ))ms after SIGTERM"
    fi
    wait "$tpid" 2>/dev/null
}

run_case plain    18190 0 127.0.0.1
run_case reload   18191 1 127.0.0.1
run_case any-addr 18192 0 0.0.0.0

[ $fail -eq 0 ] || { echo "shutdown-smoke: FAILED"; exit 1; }
echo "shutdown-smoke: OK"
