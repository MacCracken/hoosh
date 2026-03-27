#!/usr/bin/env bash
set -euo pipefail

# Run criterion benchmarks and produce two outputs:
#   1) CSV history (appended each run)   — for tracking regressions over time
#   2) Markdown table (overwritten)      — last 3 runs per benchmark for trend tracking
#
# CSV format: timestamp,commit,branch,benchmark,low_ns,estimate_ns,high_ns
#   - low_ns/estimate_ns/high_ns are the criterion confidence interval bounds
#
# Usage:
#   ./scripts/bench-history.sh                          # defaults
#   ./scripts/bench-history.sh results.csv results.md   # custom paths

HISTORY_FILE="${1:-bench-history.csv}"
MD_FILE="${2:-benchmarks.md}"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")

# Create CSV header if file doesn't exist
if [ ! -f "$HISTORY_FILE" ]; then
    echo "timestamp,commit,branch,benchmark,low_ns,estimate_ns,high_ns" > "$HISTORY_FILE"
fi

echo "╔══════════════════════════════════════════╗"
echo "║         hoosh benchmark suite            ║"
echo "╠══════════════════════════════════════════╣"
echo "║  commit: $COMMIT                          ║"
echo "║  branch: $BRANCH                            ║"
echo "║  date:   $TIMESTAMP   ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Run all benchmarks and capture output, stripping ANSI escape codes
BENCH_OUTPUT=$(cargo bench 2>&1 | sed 's/\x1b\[[0-9;]*m//g')

# Show full output
echo "$BENCH_OUTPUT"
echo ""

# ── Helper: normalise a value+unit pair to nanoseconds ───────────────────────
to_ns() {
    local val="$1" unit="$2"
    case "$unit" in
        ps)     awk "BEGIN {printf \"%.4f\", $val / 1000}" ;;
        ns)     echo "$val" ;;
        µs|us)  awk "BEGIN {printf \"%.4f\", $val * 1000}" ;;
        ms)     awk "BEGIN {printf \"%.4f\", $val * 1000000}" ;;
        s)      awk "BEGIN {printf \"%.4f\", $val * 1000000000}" ;;
        *)      echo "$val" ;;
    esac
}

# ── Parse criterion output and append to CSV ─────────────────────────────────
LINES_ADDED=0
PREV_LINE=""

while IFS= read -r line; do
    if [[ "$line" == *"time:"*"["* ]]; then
        # Extract benchmark name
        BENCH_NAME=$(echo "$line" | sed -E 's/[[:space:]]*time:.*//' | xargs)
        if [ -z "$BENCH_NAME" ]; then
            BENCH_NAME=$(echo "$PREV_LINE" | xargs)
        fi

        # Extract the three values inside brackets: [low mid high]
        VALS=$(echo "$line" | sed -E 's/.*\[(.+)\]/\1/')
        LOW_VAL=$(echo "$VALS" | awk '{print $1}')
        LOW_UNIT=$(echo "$VALS" | awk '{print $2}')
        MID_VAL=$(echo "$VALS" | awk '{print $3}')
        MID_UNIT=$(echo "$VALS" | awk '{print $4}')
        HIGH_VAL=$(echo "$VALS" | awk '{print $5}')
        HIGH_UNIT=$(echo "$VALS" | awk '{print $6}')

        LOW_NS=$(to_ns "$LOW_VAL" "$LOW_UNIT")
        MID_NS=$(to_ns "$MID_VAL" "$MID_UNIT")
        HIGH_NS=$(to_ns "$HIGH_VAL" "$HIGH_UNIT")

        # Append to CSV
        echo "${TIMESTAMP},${COMMIT},${BRANCH},${BENCH_NAME},${LOW_NS},${MID_NS},${HIGH_NS}" >> "$HISTORY_FILE"
        LINES_ADDED=$((LINES_ADDED + 1))
    fi
    PREV_LINE="$line"
done <<< "$BENCH_OUTPUT"

# ── Generate benchmarks.md with 3-point trend using python ───────────────────
python3 - "$HISTORY_FILE" "$MD_FILE" <<'PYEOF'
import csv, sys
from collections import OrderedDict

history_file = sys.argv[1]
md_file = sys.argv[2]

rows = list(csv.DictReader(open(history_file)))
if not rows:
    sys.exit(0)

timestamps = list(OrderedDict.fromkeys(r["timestamp"] for r in rows))
pick = timestamps[-3:] if len(timestamps) >= 3 else timestamps

data = OrderedDict()
commits = {}
for r in rows:
    ts = r["timestamp"]
    if ts in pick:
        data.setdefault(r["benchmark"], {})[ts] = (
            float(r["low_ns"]),
            float(r["estimate_ns"]),
            float(r["high_ns"]),
        )
        commits[ts] = r["commit"]

labels = []
for i, ts in enumerate(pick):
    if len(pick) >= 3:
        if i == 0: labels.append(f"Baseline (`{commits[ts]}`)")
        elif i == len(pick) - 1: labels.append(f"Current (`{commits[ts]}`)")
        else: labels.append(f"Previous (`{commits[ts]}`)")
    else:
        labels.append(f"{ts[:10]} (`{commits[ts]}`)")

def fmt_ns(ns):
    if ns < 1: return f"{ns*1000:.2f} ps"
    elif ns < 1000: return f"{ns:.2f} ns"
    elif ns < 1_000_000: return f"{ns/1000:.2f} us"
    elif ns < 1_000_000_000: return f"{ns/1_000_000:.2f} ms"
    else: return f"{ns/1_000_000_000:.2f} s"

def delta(old, new):
    if old == 0: return ""
    pct = ((new - old) / old) * 100
    if pct < -3: return f" **{pct:+.0f}%**"
    elif pct > 3: return f" {pct:+.0f}%"
    return ""

with open(md_file, "w") as f:
    ts_last = pick[-1]
    f.write("# Benchmarks\n\n")
    f.write(f"Latest: **{ts_last}** — commit `{commits[ts_last]}`\n\n")

    groups = OrderedDict()
    for bench in data:
        parts = bench.split("/")
        group = parts[0] if len(parts) >= 2 else "ungrouped"
        groups.setdefault(group, []).append(bench)

    for group, benches in groups.items():
        f.write(f"## {group}\n\n")
        cols = " | ".join(labels)
        f.write(f"| Benchmark | {cols} |\n")
        f.write(f"|-----------|{'|'.join(['------'] * len(labels))}|\n")
        for bench in benches:
            name = bench.split("/")[1] if "/" in bench else bench
            vals = data[bench]
            cells = []
            baseline_ns = None
            for i, ts in enumerate(pick):
                if ts not in vals: cells.append("—"); continue
                _low, est, _high = vals[ts]
                cell = fmt_ns(est)
                if i == 0: baseline_ns = est
                elif baseline_ns is not None: cell += delta(baseline_ns, est)
                cells.append(cell)
            f.write(f"| `{name}` | {' | '.join(cells)} |\n")
        f.write("\n")

    f.write("---\n\n")
    f.write("Generated by `./scripts/bench-history.sh`. Full history in `bench-history.csv`.\n")

print(f"  Generated {md_file} with {len(pick)}-point trend across {len(data)} benchmarks in {len(groups)} groups")
PYEOF

echo "════════════════════════════════════════════"
echo "  ${LINES_ADDED} benchmarks recorded"
echo "  CSV:      ${HISTORY_FILE}"
echo "  Markdown: ${MD_FILE}"
echo "════════════════════════════════════════════"
