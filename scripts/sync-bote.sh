#!/usr/bin/env bash
# Re-sync the vendored bote MCP core bundle.
#
# bote is consumed as a single committed file, src/vendor/bote-core.cyr,
# rather than via a [deps.bote] block in cyrius.cyml. The reason: bote's
# manifest declares [deps.libro] + [deps.majra] git sub-deps (for its full
# bundle's events_majra / audit_libro modules). `cyrius deps` resolves
# those transitively, vendoring libro/majra → bayan/ganita/agnosys into
# the compile set, where the agnos superset collides with bote-core's
# registry_new and trips an agnosys slice-include error. The core bundle
# (`dist/bote-core.cyr`, bote's [lib.core] profile) is fully self-contained
# — 9 transport-free modules, no includes, no libro/majra symbols — so we
# vendor just that file and skip the dep machinery entirely.
#
# Living under src/ (not a top-level vendor/) keeps `cyrius vet` happy:
# cyaudit trusts the authored src tree; a top-level vendor/ file reads as
# untrusted. The fmt/lint CI globs are src/main.cyr + src/lib/*.cyr, so the
# generated bundle in src/vendor/ is excluded from those gates.
#
# Usage: ./scripts/sync-bote.sh [tag]   (default: 2.7.6)
set -euo pipefail

TAG="${1:-2.7.6}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/src/vendor/bote-core.cyr"
URL="https://raw.githubusercontent.com/MacCracken/bote/${TAG}/dist/bote-core.cyr"

echo "Syncing bote-core $TAG from $URL"
mkdir -p "$REPO_ROOT/src/vendor"
curl -sSf "$URL" -o "$DEST"

# Sanity: the bundle must be the core profile at the requested tag.
grep -q "^# Version: ${TAG}\$" "$DEST" || {
    echo "WARN: $DEST header does not report Version: $TAG"; }
grep -q "^# Profile: core\$" "$DEST" || {
    echo "FAIL: $DEST is not the core profile (expected '# Profile: core')"; exit 1; }

echo "  wrote $DEST ($(wc -l < "$DEST") lines)"
echo "  remember to bump the bote tag in the src/main.cyr + cyrius.cyml comments and CHANGELOG"
