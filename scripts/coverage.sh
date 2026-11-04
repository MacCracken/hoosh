#!/usr/bin/env bash
# coverage.sh — symbol coverage of src/ by the test + bench suites.
#
# WHY NOT LINE COVERAGE. src/main.cyr is a program, not a library, so tests/hoosh.tcyr cannot link the
# whole gateway (its globals, threads and accept loop come with it). The suite does two things instead:
#   - it INCLUDES the src/lib modules whose own includes allow it and calls their real functions:
#     the "included for real" line this script prints lists them (18 of 37 as of 2.9.5);
#   - it MIRRORS the rest, re-implementing the logic under test with stdlib-only code.
# `cyrius coverage` reports on the vendored stdlib rather than this repo (filed upstream), so there is no
# line-coverage number. rust-old's codecov gate (project 85% / patch 80%) has no like-for-like port.
#
# WHAT THIS MEASURES INSTEAD. For every function defined in src/, is that name referenced anywhere in
# tests/? A function nobody wrote a test, a call or a mirror for is invisible to the suite, and adding
# one to src/ without touching tests/ moves the number down. It is a coverage FLOOR, not a proof.
#
# KNOWN LIMITATIONS, stated plainly:
#   - It reads NAMES, not execution, and that cuts both ways. A real module's internal helper that runs
#     only through its caller (compression.cyr's _cmp_is_role, say) counts as uncovered; a mirror that
#     reuses a src function's name used to count as covering it. 2.9.4 saw the number dip 306 -> 301
#     when such same-name copies were deleted, though the real functions had just started running.
#   - For a MIRRORED module it cannot catch drift: src and its copy diverging while both stay internally
#     consistent. That has bitten more than once (2.5.6 pricing, 2.5.7 audit linkage, and the drifted
#     copies 2.9.3 / 2.9.4 deleted), which is why modules move to "included for real" one by one. A
#     module included for real cannot drift from its tests.
# Treat a high number here as "nothing is unwatched", not "everything is correct".
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

# Informational: which src/lib modules the test suite includes for real (their functions run, rather than
# a copy of them). Not part of the gate.
REAL=$(grep -oE '^include "src/lib/[a-z_]+\.cyr"' tests/hoosh.tcyr 2>/dev/null \
    | sed -E 's#.*/([a-z_]+)\.cyr"#\1#' | sort -u)
NREAL=$(printf '%s\n' "$REAL" | grep -c .)
NMOD=$(ls src/lib/*.cyr 2>/dev/null | wc -l)

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
echo "  included for real: ${NREAL} of ${NMOD} src/lib modules ($(printf '%s ' $REAL | sed 's/ $//'))"

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
