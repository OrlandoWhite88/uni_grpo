#!/usr/bin/env bash
set -euo pipefail

# Startup script for this repo.
# It will:
#   1) sudo apt update
#   2) sudo apt install -y python3 python3-pip snapd
#   3) ensure /snap/bin is available on PATH
#   4) install astral-uv via snap if it is not already installed
#   5) create (or reuse) a uv-managed virtual environment in .venv
#   6) activate the virtual environment
#   7) install requirements with uv pip install -r requirements.txt
#   8) run python download_dataset.py
#
# Usage after cloning:
#   bash startup.sh
# Or:
#   chmod +x startup.sh && ./startup.sh
#
# After the script finishes you can reactivate the environment with:
#   source .venv/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[1/8] sudo apt update"
sudo apt update

echo "[2/8] Install python3, pip, and snapd"
sudo apt install -y python3 python3-pip snapd

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found after installation. Please verify your apt sources."
  exit 1
fi

if ! command -v pip3 >/dev/null 2>&1; then
  echo "pip3 not found after installation. Please verify your apt sources."
  exit 1
fi

echo "[3/8] Ensure /snap/bin is on PATH"
export PATH="/snap/bin:$PATH"

if ! command -v snap >/dev/null 2>&1; then
  echo "snap command not available. You may need to log out/in or enable snapd manually."
  exit 1
fi

echo "[4/8] Ensure astral-uv snap is installed"
if snap list 2>/dev/null | grep -Eq "^astral-uv\s"; then
  echo "astral-uv snap already installed."
else
  sudo snap install astral-uv --classic
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv command not available even after installing the snap. Check snap permissions or PATH."
  exit 1
fi

echo "[5/8] Create uv virtual environment (.venv)"
uv venv

echo "[6/8] Activate virtual environment"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[7/8] Install Python dependencies via uv"
uv pip install -r requirements.txt

echo "[8/8] Download dataset"
python download_dataset.py

echo "Setup complete. To use this environment later run: source .venv/bin/activate"
