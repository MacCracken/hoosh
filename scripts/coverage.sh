#!/usr/bin/env bash
# coverage.sh — symbol coverage of src/ by the test + bench suites.
#
# WHY NOT LINE COVERAGE. hoosh's tests are a self-contained MIRROR: tests/
# hoosh.tcyr re-implements the logic under test with stdlib-only includes rather
# than linking src/ (src/main.cyr is a program, not a library, and pulling it in
# drags the whole gateway including its globals and threads). So no test ever
# executes a line of src/, and a line-coverage number would be a flat 0% — which
# is why `cyrius coverage` reports on the vendored stdlib instead. rust-old's
# codecov gate (project 85% / patch 80%) does not have a like-for-like port.
#
# WHAT THIS MEASURES INSTEAD. For every function defined in src/, is that name
# referenced anywhere in tests/? That is the property the mirror design can
# actually hold: a function nobody wrote a test or a mirror for is invisible to
# the suite, and adding one to src/ without touching tests/ moves the number
# down. It is a coverage FLOOR, not a proof.
#
# KNOWN LIMITATION, stated plainly: because the mirror re-implements rather than
# calls, this cannot catch mirror DRIFT — src and its mirror diverging while
# both are internally consistent. That failure mode is real and has bitten twice
# (2.5.6 pricing local-first, 2.5.7 audit chain linkage: in both cases the
# mirror was right and src was wrong, and the suite stayed green). Treat a high
# number here as "nothing is unwatched", not "everything is correct".
#
# Usage: scripts/coverage.sh [min_percent]   (default 80)
# NB: deliberately NOT `set -o pipefail`. The membership check below is a plain
# grep against a file, but an earlier version piped into `grep -q`, which exits
# on the first match and SIGPIPEs the writer — with pipefail that reads as a
# failed pipeline, so every symbol scored as uncovered and the gate reported 0%.
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
MIN="${1:-80}"

SRC_FILES=$(ls src/lib/*.cyr src/main.cyr 2>/dev/null)

# Every identifier the suites mention, one per line, deduped. Building this once
# turns ~500 scans of a 270 KB blob into ~500 lookups in a small sorted file.
SYMS=$(mktemp)
trap 'rm -f "$SYMS"' EXIT
cat tests/hoosh.tcyr tests/hoosh.bcyr 2>/dev/null \
    | grep -oE '[a-zA-Z_][a-zA-Z0-9_]*' | sort -u > "$SYMS"

total=0
covered=0
uncovered=""

for f in $SRC_FILES; do
    # Function names defined in this file.
    while IFS= read -r fname; do
        [ -z "$fname" ] && continue
        total=$((total + 1))
        if grep -qxF "$fname" "$SYMS"; then
            covered=$((covered + 1))
        else
            uncovered="${uncovered}${f#src/}:${fname}"$'\n'
        fi
    done < <(grep -oE '^fn [a-zA-Z_][a-zA-Z0-9_]*' "$f" | sed 's/^fn //')
done

if [ "$total" -eq 0 ]; then
    echo "coverage: no functions found in src/ — is this the repo root?"
    exit 1
fi

pct=$(( covered * 100 / total ))

echo "=== hoosh symbol coverage (src/ referenced by tests/) ==="
echo "  covered: ${covered}/${total} functions (${pct}%)"
echo "  minimum: ${MIN}%"

if [ -n "${VERBOSE:-}" ]; then
    echo ""
    echo "--- unreferenced ---"
    printf '%s' "$uncovered" | sed 's/^/  /'
fi

if [ "$pct" -lt "$MIN" ]; then
    echo ""
    echo "FAIL: symbol coverage ${pct}% is below the ${MIN}% floor."
    echo "      Run with VERBOSE=1 to list unreferenced functions."
    exit 1
fi

echo "  OK"
exit 0
