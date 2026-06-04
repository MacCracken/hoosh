#!/usr/bin/env bash
# Version bump for hoosh — VERSION is the single source of truth.
# cyrius.cyml carries `version = "${file:VERSION}"`, so the manifest
# tracks the bump automatically (nothing to edit there).
#
# hoosh is a server binary — it publishes no distlib bundle, so there
# is no dist/ to regenerate. This script:
#   1. writes VERSION
#   2. updates the `- **Version**:` line in CLAUDE.md
#   3. bumps the HOOSH_VERSION constant in src/lib/types.cyr so the
#      running binary reports the right version
#   4. adds a CHANGELOG.md stub for the new version (if missing)
#
# Tag and push after bumping.
set -euo pipefail

[ $# -ne 1 ] && echo "Usage: $0 <version>  (current: $(cat VERSION))" && exit 1
NEW="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
OLD="$(tr -d '[:space:]' < VERSION)"

# 1. VERSION file (source of truth)
echo "$NEW" > VERSION
echo "  VERSION: $OLD -> $NEW"

# 2. CLAUDE.md `- **Version**:` line
if [ -f CLAUDE.md ] && grep -q '^- \*\*Version\*\*:' CLAUDE.md; then
    sed -i "s/^- \*\*Version\*\*:.*/- **Version**: SemVer ${NEW} stable/" CLAUDE.md
    echo "  CLAUDE.md version line updated"
fi

# 3. HOOSH_VERSION constant in source
if [ -f src/lib/types.cyr ]; then
    sed -i "s/^var HOOSH_VERSION = \".*\";/var HOOSH_VERSION = \"${NEW}\";/" src/lib/types.cyr
    echo "  src/lib/types.cyr HOOSH_VERSION updated"
fi

# 4. CHANGELOG.md — add a stub if there is no entry for $NEW yet
if [ -f CHANGELOG.md ] && ! grep -q "## \[$NEW\]" CHANGELOG.md; then
    TODAY=$(date -u +%Y-%m-%d)
    awk -v v="$NEW" -v d="$TODAY" '
        /^## \[/ && !done {
            print "## [" v "] — " d "\n\n**TODO:** describe this release.\n"
            done = 1
        }
        { print }
    ' CHANGELOG.md > CHANGELOG.md.tmp && mv CHANGELOG.md.tmp CHANGELOG.md
    echo "  CHANGELOG.md stub added for $NEW"
fi

echo "Bumped to ${NEW}. Fill in CHANGELOG.md, run ./scripts/bench-history.sh, then tag and push."
