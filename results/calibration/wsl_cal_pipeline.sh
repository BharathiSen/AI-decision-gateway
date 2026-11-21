#!/bin/bash
set -euo pipefail
export SUDO_ASKPASS=/tmp/verifix_askpass.sh
cd /home/bharathi/AI-decision-gateway
mkdir -p results/calibration

if ! command -v iperf3 >/dev/null 2>&1; then
  echo "iperf3 missing; install before calibration" >&2
  exit 2
fi
echo "iperf3: $(command -v iperf3)"

count_t10() {
  local noise="$1"
  PYTHONPATH=. python3 results/calibration/_count_t10.py "$noise"
}

for noise in N0 N1 N2; do
  echo "=== calibrate ${noise} (with iperf3) ==="
  while true; do
    sudo -A mn -c || true
    sudo -A env PYTHONPATH=. python3 -u bench/calibrate_healthy.py \
      --noise "${noise}" --runs 200 --eval-runs 50 --batch 250 --fill-speed \
      --samples results/calibration_samples_episode.jsonl \
      --report "results/calibration/wsl-${noise}.json" || true
    n=$(count_t10 "${noise}")
    echo "${noise} usable-T10 samples=${n}"
    if [ "${n}" -ge 250 ]; then
      break
    fi
    echo "${noise} incomplete; continuing"
  done
done

echo "=== reprint + write-thresholds ==="
PYTHONPATH=. python3 bench/calibrate_healthy.py \
  --from-saved --noise all --runs 200 --eval-runs 50 \
  --samples results/calibration_samples_episode.jsonl \
  --report results/calibration/wsl-all.json

sudo -A env PYTHONPATH=. python3 bench/calibrate_healthy.py \
  --from-saved --noise all --runs 200 --eval-runs 50 \
  --samples results/calibration_samples_episode.jsonl \
  --report results/calibration/wsl-all.json \
  --write-thresholds

PYTHONPATH=. python3 - <<'PY'
from bench.outcomes import threshold_status
print({n: threshold_status(n).get("status") for n in ("N0", "N1", "N2")})
PY

echo "=== live-accept ==="
sudo -A env PYTHONPATH=. python3 bench/accept.py --live-accept
echo PIPELINE_DONE
read -r -p "Press Enter to close..."
