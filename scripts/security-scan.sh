#!/usr/bin/env bash
# security-scan.sh — lightweight security-pattern scan over hoosh's OWN source
# (src/main.cyr + src/lib/*.cyr). Vendored bundles + the cyrius stdlib are out of
# scope (audited upstream). Exits non-zero on any hit so CI gates on it. Kept
# low-false-positive: security issues, not style. (TLS-backend safety is covered
# by the main.cyr startup warning + CLAUDE.md, not here.)
set -u

SRC="src/main.cyr src/lib/*.cyr"
fail=0

flag() {  # <description> <grep-ERE-pattern>
    local desc="$1" pat="$2"
    local hits
    hits=$(grep -nE "$pat" $SRC 2>/dev/null) || true
    if [ -n "$hits" ]; then
        echo "✗ $desc:"; echo "$hits" | sed 's/^/    /'
        fail=1
    else
        echo "✓ $desc"
    fi
}

echo "== hoosh security-pattern scan =="

# 1. hoosh's own code must not shell out — subprocess execution lives in the
#    vendored ai-hwaccel / cyrius stdlib, never here (command-injection surface).
#    Word boundaries so e.g. `_obj_is_system` is not a hit.
flag "no subprocess exec in src" '\b(execve|popen|sys_fork)\b|\bsystem\(|/bin/sh'

# 2. No hardcoded absolute config/secret paths (config is hoosh.cyml + $ENV).
flag "no hardcoded /etc paths" '"/etc/'

# 3. No obvious hardcoded secrets — real provider keys use $ENV expansion.
flag "no hardcoded provider keys" '"(sk-[A-Za-z0-9]{8}|AKIA[A-Z0-9]{12}|ghp_[A-Za-z0-9]{8})'

if [ $fail -ne 0 ]; then echo "SECURITY SCAN FAILED"; exit 1; fi
echo "security scan clean"
