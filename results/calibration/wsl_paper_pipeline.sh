#!/bin/bash
set -euo pipefail
export SUDO_ASKPASS=/tmp/verifix_askpass.sh
cd /home/bharathi/AI-decision-gateway
mkdir -p results/pilot results/paper results/calibration
LOG=results/calibration/paper_v1_run.log
exec > >(tee -a "$LOG") 2>&1

if [ -z "${GROQ_API_KEY:-}" ]; then
  echo "GROQ_API_KEY unset; paper batch requires --require-model" >&2
  exit 2
fi

echo "=== paper_v1 start $(date -Is) ==="
echo "GROQ_PRESENT=yes"

PYTHONPATH=. python3 - <<'PY'
import json
from pathlib import Path
from bench.outcomes import threshold_status
p = Path("results/paper/paper_v1_manifest.json")
m = json.loads(p.read_text(encoding="utf-8"))
print("freeze_id", m.get("freeze_id"))
print("live_host", m.get("live_host"))
print("episodes", m.get("target_episodes"))
print("thresholds", (m.get("artifacts") or {}).get("thresholds.yaml", {}).get("sha256", "")[:16])
print("arms", m.get("paper_arms"))
print("threshold_status", {n: threshold_status(n).get("status") for n in ("N0", "N1", "N2")})
PY

echo "=== mn -c ==="
sudo -A mn -c || true

echo "=== paper_v1 live 300 stratified ==="
sudo -A --preserve-env=GROQ_API_KEY env PYTHONPATH=. python3 -u scripts/finish_evaluation.py \
  --mode live --run-name paper_v1 --episodes 300 --paper-stratified

echo "PAPER_PIPELINE_DONE $(date -Is)"
if [ -t 0 ] && [ "${NONINTERACTIVE:-0}" != "1" ]; then
  read -r -p "Press Enter to close..."
fi
