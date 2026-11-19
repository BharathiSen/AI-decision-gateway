#!/bin/bash
set -euo pipefail
export SUDO_ASKPASS=/tmp/verifix_askpass.sh
cd /home/bharathi/AI-decision-gateway
mkdir -p results/calibration

echo "=== waiting for current N0 job if any ==="
while pgrep -f 'calibrate_healthy.py --noise N0' >/dev/null 2>&1; do
  n=$(wc -l < results/calibration_samples_episode.jsonl | tr -d ' ')
  echo "N0 still running; samples=${n}; $(date -Is)"
  sleep 60
done
n=$(wc -l < results/calibration_samples_episode.jsonl | tr -d ' ')
echo "N0 process gone; samples=${n}"

if [ "${n}" -lt 250 ]; then
  echo "=== resume N0 to 250 ==="
  sudo -A mn -c || true
  sudo -A env PYTHONPATH=. python3 -u bench/calibrate_healthy.py \
    --noise N0 --runs 200 --eval-runs 50 --batch 200 --fill-speed \
    --samples results/calibration_samples_episode.jsonl \
    --report results/calibration/wsl-n0.json
fi

for noise in N1 N2; do
  echo "=== calibrate ${noise} ==="
  sudo -A mn -c || true
  sudo -A env PYTHONPATH=. python3 -u bench/calibrate_healthy.py \
    --noise "${noise}" --runs 200 --eval-runs 50 --batch 250 --fill-speed \
    --samples results/calibration_samples_episode.jsonl \
    --report "results/calibration/wsl-${noise}.json"
done

echo "=== from-saved reprint all ==="
PYTHONPATH=. python3 bench/calibrate_healthy.py \
  --from-saved --noise all --runs 200 --eval-runs 50 \
  --samples results/calibration_samples_episode.jsonl \
  --report results/calibration/wsl-all.json

echo "=== write-thresholds if ready ==="
sudo -A env PYTHONPATH=. python3 bench/calibrate_healthy.py \
  --from-saved --noise all --runs 200 --eval-runs 50 \
  --samples results/calibration_samples_episode.jsonl \
  --report results/calibration/wsl-all.json \
  --write-thresholds || true

echo "=== threshold status ==="
PYTHONPATH=. python3 - <<'PY'
from bench.outcomes import threshold_status
print({n: threshold_status(n).get("status") for n in ("N0", "N1", "N2")})
PY

echo "=== live-accept ==="
sudo -A env PYTHONPATH=. python3 bench/accept.py --live-accept
echo PIPELINE_DONE
read -r -p "Press Enter to close..."
