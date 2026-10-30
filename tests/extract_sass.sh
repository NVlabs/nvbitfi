#!/bin/bash
# Print the SASS of a CUDA binary or library, one instruction per line, in the
# format read by dests_test:   tests/extract_sass.sh app [sm_120] | tests/dests_test
# Needs cuobjdump (CUDA toolkit) in PATH.
set -e
BIN=${1:?usage: extract_sass.sh <binary|library> [sm_XX]}
ARCH=${2:-sm_120}
cuobjdump -sass -arch "$ARCH" "$BIN" \
	| grep -E '^\s+/\*[0-9a-f]{4,}\*/' \
	| sed -E 's#^\s+/\*[0-9a-f]+\*/\s+##; s#\s*;.*##'
