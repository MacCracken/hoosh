#!/bin/sh
CC="${1:-./build/cc3}"
echo "=== hoosh tests ==="
cat src/main.cyr | "$CC" > /tmp/hoosh_test && chmod +x /tmp/hoosh_test && /tmp/hoosh_test
echo "exit: $?"
rm -f /tmp/hoosh_test
