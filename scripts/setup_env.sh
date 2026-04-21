#!/usr/bin/env bash
# MDMA environment installer (Linux / macOS / WSL).
#
# Delegates to scripts/setup_env.py so the real logic stays in one
# place. Any extra args are forwarded — e.g.:
#   ./scripts/setup_env.sh --profile full --with-dev

set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "error: $PYTHON_BIN not on PATH. Set PYTHON=/path/to/python3 or install Python 3.9+." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$PYTHON_BIN" "$SCRIPT_DIR/setup_env.py" "$@"
