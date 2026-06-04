#!/usr/bin/env bash
set -euo pipefail

# Run the Cyrius benchmark suite (tests/*.bcyr) and produce two outputs:
#   1) CSV history (appended each run)   — for tracking regressions over time
#   2) Markdown table (overwritten)      — last 3 runs per benchmark, trend
#
# CSV format: timestamp,commit,branch,suite,benchmark,avg_ns,min_ns,max_ns,iters
#   - parsed from `cyrius bench` lines of the form:
#       <name>: <avg>ns avg (min=<min>ns max=<max>ns) [<n> iters]
#
# Builds with the pinned toolchain (~/.cyrius/versions/<pin>/bin/cyrius)
# when present so numbers match CI; falls back to the PATH `cyrius`.
#
# Usage:
#   ./scripts/bench-history.sh                          # defaults
#   ./scripts/bench-history.sh results.csv results.md   # custom paths

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

HISTORY_FILE="${1:-bench-history.csv}"
MD_FILE="${2:-benchmarks.md}"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")

# Resolve the pinned cyrius (CI parity), else PATH cyrius.
PIN="$(grep -E '^cyrius[[:space:]]*=' cyrius.cyml | sed -E 's/.*"([^"]+)".*/\1/')"
PINNED_BIN="$HOME/.cyrius/versions/${PIN}/bin/cyrius"
if [ -x "$PINNED_BIN" ]; then CYBIN="$PINNED_BIN"; else CYBIN="cyrius"; fi
export CYRIUS_NO_WARN_PIN_DRIFT=1

if [ ! -f "$HISTORY_FILE" ]; then
    echo "timestamp,commit,branch,suite,benchmark,avg_ns,min_ns,max_ns,iters" > "$HISTORY_FILE"
fi

echo "╔══════════════════════════════════════════╗"
echo "║          hoosh benchmark suite           ║"
echo "╠══════════════════════════════════════════╣"
echo "║  commit: $COMMIT"
echo "║  branch: $BRANCH"
echo "║  cyrius: $("$CYBIN" --version 2>/dev/null || echo unknown) (pin ${PIN})"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── Helper: normalise a numeric value + unit to nanoseconds ──────────────────
to_ns() {
    awk -v v="$1" -v u="$2" 'BEGIN{
        if (u=="ps") printf "%.4f", v/1000;
        else if (u=="ns") printf "%s", v;
        else if (u=="us"||u=="µs") printf "%.4f", v*1000;
        else if (u=="ms") printf "%.4f", v*1000000;
        else if (u=="s") printf "%.4f", v*1000000000;
        else printf "%s", v;
    }'
}

LINES_ADDED=0
for f in tests/*.bcyr; do
    [ -f "$f" ] || continue
    suite=$(basename "$f" .bcyr)
    echo "--- $suite ---"
    "$CYBIN" build "$f" "/tmp/bench_${suite}" >/dev/null 2>&1
    result=$("/tmp/bench_${suite}" 2>&1)
    echo "$result"
    echo ""

    # Parse "<name>: <avg><unit> avg (min=<min><unit> max=<max><unit>) [<n> iters]"
    while IFS= read -r line; do
        echo "$line" | grep -qE '^[[:space:]]*[A-Za-z0-9_]+:[[:space:]]*[0-9]' || continue
        bname=$(echo "$line"  | sed -E 's/^[[:space:]]*([A-Za-z0-9_]+):.*/\1/')
        avg_v=$(echo "$line"  | sed -E 's/.*:[[:space:]]*([0-9.]+)([a-zµ]+) avg.*/\1/')
        avg_u=$(echo "$line"  | sed -E 's/.*:[[:space:]]*([0-9.]+)([a-zµ]+) avg.*/\2/')
        min_v=$(echo "$line"  | sed -E 's/.*min=([0-9.]+)([a-zµ]+).*/\1/')
        min_u=$(echo "$line"  | sed -E 's/.*min=([0-9.]+)([a-zµ]+).*/\2/')
        max_v=$(echo "$line"  | sed -E 's/.*max=([0-9.]+)([a-zµ]+).*/\1/')
        max_u=$(echo "$line"  | sed -E 's/.*max=([0-9.]+)([a-zµ]+).*/\2/')
        iters=$(echo "$line"  | sed -E 's/.*\[([0-9]+) iters\].*/\1/')
        echo "${TIMESTAMP},${COMMIT},${BRANCH},${suite},${bname},$(to_ns "$avg_v" "$avg_u"),$(to_ns "$min_v" "$min_u"),$(to_ns "$max_v" "$max_u"),${iters}" >> "$HISTORY_FILE"
        LINES_ADDED=$((LINES_ADDED + 1))
    done <<< "$result"
done

# ── Generate markdown trend (last 3 runs per benchmark, avg_ns + delta) ───────
python3 - "$HISTORY_FILE" "$MD_FILE" <<'PYEOF'
import csv, sys
from collections import OrderedDict

rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    sys.exit(0)

timestamps = list(OrderedDict.fromkeys(r["timestamp"] for r in rows))
pick = timestamps[-3:]
data, commits = OrderedDict(), {}
for r in rows:
    if r["timestamp"] in pick:
        key = f'{r["suite"]}/{r["benchmark"]}'
        data.setdefault(key, {})[r["timestamp"]] = float(r["avg_ns"])
        commits[r["timestamp"]] = r["commit"]

labels = []
for i, ts in enumerate(pick):
    role = "Baseline" if i == 0 and len(pick) >= 3 else ("Current" if i == len(pick) - 1 else "Previous")
    labels.append(f"{role} (`{commits[ts]}`)")

def fmt(ns):
    if ns < 1000: return f"{ns:.0f} ns"
    if ns < 1_000_000: return f"{ns/1000:.2f} us"
    if ns < 1_000_000_000: return f"{ns/1_000_000:.2f} ms"
    return f"{ns/1_000_000_000:.2f} s"

def delta(old, new):
    if not old: return ""
    pct = (new - old) / old * 100
    if pct < -3: return f" **{pct:+.0f}%**"
    if pct > 3:  return f" {pct:+.0f}%"
    return ""

with open(sys.argv[2], "w") as f:
    f.write(f"# Benchmarks\n\nLatest: **{pick[-1]}** — commit `{commits[pick[-1]]}`\n\n")
    groups = OrderedDict()
    for key in data:
        groups.setdefault(key.split("/")[0], []).append(key)
    for group, benches in groups.items():
        f.write(f"## {group}\n\n| Benchmark | {' | '.join(labels)} |\n")
        f.write(f"|-----------|{'|'.join(['------'] * len(labels))}|\n")
        for key in benches:
            name = key.split("/", 1)[1]
            cells, base = [], None
            for i, ts in enumerate(pick):
                v = data[key].get(ts)
                if v is None:
                    cells.append("—"); continue
                cell = fmt(v)
                if i == 0: base = v
                elif base is not None: cell += delta(base, v)
                cells.append(cell)
            f.write(f"| `{name}` | {' | '.join(cells)} |\n")
        f.write("\n")
    f.write("---\n\nGenerated by `./scripts/bench-history.sh`. Full history in `bench-history.csv`.\n")
print(f"  markdown trend -> {sys.argv[2]}")
PYEOF

echo "════════════════════════════════════════════"
echo "  ${LINES_ADDED} benchmarks recorded"
echo "  CSV:      ${HISTORY_FILE}"
echo "  Markdown: ${MD_FILE}"
echo "════════════════════════════════════════════"
