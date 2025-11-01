#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements.txt

python - <<'PY'
import torch, platform
print("Torch:", torch.__version__, "CUDA avail:", torch.cuda.is_available(), "Py:", platform.python_version())
PY
