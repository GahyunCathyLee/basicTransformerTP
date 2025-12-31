#!/usr/bin/env bash
# scripts/run_ablations_continue.sh
# - Runs multiple ablation configs sequentially
# - Continues even if one experiment fails
# - Saves per-experiment logs under logs/
# - (Optional) Skips experiments if ckpt_dir/best.pt already exists

set -u  # do NOT use -e (we want to continue on failures)
set -o pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Run this script from tp_baseline/ (project root)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs

# List only ablations (exclude full baseline as requested)
CONFIGS=(
  "configs/ablation_ego_only.yaml"
  "configs/ablation_no_context.yaml"
  "configs/ablation_no_safety.yaml"
  "configs/ablation_no_static.yaml"
  "configs/ablation_no_slot_emb.yaml"
)

# Toggle: set to 1 to skip configs that already have ckpt_dir/best.pt
SKIP_IF_BEST_EXISTS=${SKIP_IF_BEST_EXISTS:-1}

get_ckpt_dir() {
  local cfg="$1"
  python3 - << 'PY' "$cfg"
import sys, yaml
cfg_path = sys.argv[1]
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)
print(cfg.get("train", {}).get("ckpt_dir", "ckpts/unknown"))
PY
}

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[$(timestamp)] Starting ablation runs"
echo "  ROOT_DIR=$ROOT_DIR"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  SKIP_IF_BEST_EXISTS=$SKIP_IF_BEST_EXISTS"
echo

FAILED=0
TOTAL=0

for cfg in "${CONFIGS[@]}"; do
  TOTAL=$((TOTAL+1))
  name="$(basename "$cfg" .yaml)"

  if [[ ! -f "$cfg" ]]; then
    echo "[$(timestamp)] [SKIP] Config not found: $cfg"
    continue
  fi

  ckpt_dir="$(get_ckpt_dir "$cfg")"
  # ckpt_dir in yaml may be relative; resolve relative to project root
  if [[ "$ckpt_dir" != /* ]]; then
    ckpt_dir="$ROOT_DIR/$ckpt_dir"
  fi

  if [[ "$SKIP_IF_BEST_EXISTS" == "1" && -f "$ckpt_dir/best.pt" ]]; then
    echo "[$(timestamp)] [SKIP] $name (best.pt exists): $ckpt_dir/best.pt"
    continue
  fi

  log_path="logs/${name}.log"
  echo "[$(timestamp)] [RUN ] $name"
  echo "         cfg: $cfg"
  echo "     ckpt_dir: $ckpt_dir"
  echo "         log: $log_path"

  # Run training; keep going even if it fails
  python3 -m src.train --config "$cfg" 2>&1 | tee "$log_path"
  status=${PIPESTATUS[0]}

  if [[ $status -ne 0 ]]; then
    echo "[$(timestamp)] [FAIL] $name (exit=$status)"
    FAILED=$((FAILED+1))
  else
    echo "[$(timestamp)] [OK  ] $name"
  fi
  echo
done

echo "[$(timestamp)] Finished."
echo "  total configs considered: $TOTAL"
echo "  failed: $FAILED"
echo "  logs: $ROOT_DIR/logs/"
exit 0
