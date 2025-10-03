#!/usr/bin/env bash
set -euo pipefail

# Startup script for this repo.
# It will:
#   1) sudo apt update
#   2) pip install uv
#   3) uv install requirements.txt (falls back to: uv pip install -r requirements.txt)
#   4) python3 download_dataset.py
#
# Usage after cloning:
#   bash startup.sh
# Or:
#   chmod +x startup.sh && ./startup.sh

# Move to the directory containing this script (repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[1/4] sudo apt update"
sudo apt update

echo "[2/4] pip install uv"
# Prefer pip, then pip3. If neither is present, instruct how to install.
if command -v pip >/dev/null 2>&1; then
  PIP_CMD="pip"
elif command -v pip3 >/dev/null 2>&1; then
  PIP_CMD="pip3"
else
  echo "pip/pip3 not found. Install it with: sudo apt install -y python3-pip"
  exit 1
fi

# Install uv for current user to avoid requiring sudo
$PIP_CMD install --user -U uv

# Ensure ~/.local/bin is on PATH so the 'uv' command resolves (for --user installs)
export PATH="$HOME/.local/bin:$PATH"

echo "[3/4] uv install requirements.txt"
# Try requested command first; if it fails, fall back to the common syntax
if uv install requirements.txt; then
  true
else
  echo "Command 'uv install requirements.txt' failed. Falling back to: uv pip install -r requirements.txt"
  uv pip install -r requirements.txt
fi

echo "[4/4] python3 download_dataset.py"
python3 download_dataset.py

echo "Setup complete."
