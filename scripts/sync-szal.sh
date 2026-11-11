#!/usr/bin/env bash
# Re-sync the vendored szál MCP tools bundle (2.12.1).
#
# szál's tools are consumed as one committed file, src/vendor/szal-mcp.cyr: szál's `[lib.mcp]` profile, its 54 MCP
# tools with none of bote-core, majra, ai-hwaccel or the stdlib inside (hoosh already has all four). Like bote-core
# (scripts/sync-bote.sh), it is vendored rather than pulled through a [deps.szal] block, because szál's manifest
# declares bote, majra and ai-hwaccel as git sub-deps, which `cyrius deps` would resolve into the compile set beside
# hoosh's own copies. src/main.cyr includes it after bote-core and majra, as its header requires.
#
# Before vendoring a new tag, run szál's own checks against this checkout:
#   (cd ../szal && ./scripts/consumer-check.sh ../hoosh)
# It scans the bundle's names against hoosh's compile set, builds hoosh with the bundle, and registers the 54 tools
# into a dispatcher that already holds one. Then bump the tag here, in src/lib/mcp.cyr's header and in the CHANGELOG.
#
# Usage: ./scripts/sync-szal.sh [tag]   (default: 2.3.3)
set -euo pipefail

TAG="${1:-2.3.3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/src/vendor/szal-mcp.cyr"
URL="https://raw.githubusercontent.com/MacCracken/szal/${TAG}/dist/szal-mcp.cyr"

echo "Syncing szal-mcp $TAG from $URL"
mkdir -p "$REPO_ROOT/src/vendor"
curl -sSf "$URL" -o "$DEST"

# Sanity: the bundle must be the mcp profile at the requested tag.
grep -q "^# Version: ${TAG}\$" "$DEST" || {
    echo "WARN: $DEST header does not report Version: $TAG"; }
grep -q "^# Profile: mcp\$" "$DEST" || {
    echo "FAIL: $DEST is not the mcp profile (expected '# Profile: mcp')"; exit 1; }

echo "  wrote $DEST ($(wc -l < "$DEST") lines)"
echo "  remember: the tag in src/lib/mcp.cyr's header and the CHANGELOG, and any stdlib leaf its"
echo "  '# Requires' block names that cyrius.cyml's [deps].stdlib lacks"
