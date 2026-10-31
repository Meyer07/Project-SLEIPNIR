#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

find . -type f \
  ! -path './checksums.sha256' \
  ! -path './.venv/*' \
  ! -path './*/__pycache__/*' \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum > checksums.sha256

echo "Wrote $ROOT_DIR/checksums.sha256"
echo "Verify later with: sha256sum -c checksums.sha256"
