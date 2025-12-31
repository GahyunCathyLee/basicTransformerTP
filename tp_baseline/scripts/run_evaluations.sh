#!/usr/bin/env bash
# scripts/run_evaluations.sh
# Robust evaluation runner:
# - Continues even if one evaluation fails
# - Finds best.pt robustly even if ckpt_dir is ../ckpts/... or ckpts/... or configs/... etc.
# - Evaluates ablation configs only (baseline excluded)
# - Uses: python3 -m scripts.eval

set -u
set -o pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs results/json

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

# ---- ablation configs (baseline excluded) ----
CONFIGS=(
  "configs/ablation_ego_only.yaml"
  "configs/ablation_no_context.yaml"
  "configs/ablation_no_safety.yaml"
  "configs/ablation_no_static.yaml"
  "configs/ablation_no_slot_emb.yaml"
)

# Optional: skip evaluation if exp already in results.csv
SKIP_IF_ALREADY_IN_CSV=${SKIP_IF_ALREADY_IN_CSV:-0}

already_in_csv() {
  local exp_name="$1"
  local csv_path="results/results.csv"
  [[ -f "$csv_path" ]] && grep -q ",${exp_name}," "$csv_path"
}

get_ckpt_dir_from_config() {
  local cfg="$1"
  python3 - << 'PY' "$cfg"
import sys, yaml
from pathlib import Path
cfg_path = Path(sys.argv[1])
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)
print(cfg.get("train", {}).get("ckpt_dir", ""))
PY
}

# Try multiple candidate locations for best.pt
find_best_ckpt() {
  local cfg_path="$1"      # e.g., configs/ablation_no_safety.yaml
  local ckpt_dir_raw="$2"  # from yaml, may be relative weird

  local cfg_abs="$ROOT_DIR/$cfg_path"
  local cfg_dir
  cfg_dir="$(cd "$(dirname "$cfg_abs")" && pwd)"

  # Build candidate ckpt_dir interpretations
  local candidates=()

  # 1) interpret relative to project root
  if [[ "$ckpt_dir_raw" != /* ]]; then
    candidates+=("$ROOT_DIR/$ckpt_dir_raw")
  else
    candidates+=("$ckpt_dir_raw")
  fi

  # 2) interpret relative to config directory (often correct if you store configs elsewhere)
  if [[ "$ckpt_dir_raw" != /* ]]; then
    candidates+=("$cfg_dir/$ckpt_dir_raw")
  fi

  # 3) common fallback: ckpts/<exp_name>
  local name
  name="$(basename "$cfg_path" .yaml)"
  candidates+=("$ROOT_DIR/ckpts/$name")

  # 4) if ckpt_dir_raw starts with ../ckpts, also try stripping ../
  if [[ "$ckpt_dir_raw" == ../ckpts/* ]]; then
    candidates+=("$ROOT_DIR/ckpts/${ckpt_dir_raw#../ckpts/}")
  fi

  # Check each candidate
  local d
  for d in "${candidates[@]}"; do
    # normalize
    d="$(cd "$(dirname "$d")" 2>/dev/null && pwd)/$(basename "$d")" || true
    if [[ -f "$d/best.pt" ]]; then
      echo "$d/best.pt"
      return 0
    fi
  done

  # last resort: search within ROOT_DIR/ckpts for matching folder
  # (safe & limited)
  local hit
  hit="$(find "$ROOT_DIR/ckpts" -maxdepth 2 -type f -name best.pt 2>/dev/null | grep "/$name/" | head -n 1 || true)"
  if [[ -n "$hit" ]]; then
    echo "$hit"
    return 0
  fi

  echo ""
  return 1
}

FAILED=0
TOTAL=0

echo "[$(timestamp)] Starting evaluations (split=test)"
echo "  ROOT_DIR=$ROOT_DIR"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  SKIP_IF_ALREADY_IN_CSV=$SKIP_IF_ALREADY_IN_CSV"
echo

for cfg in "${CONFIGS[@]}"; do
  TOTAL=$((TOTAL+1))
  name="$(basename "$cfg" .yaml)"

  if [[ ! -f "$cfg" ]]; then
    echo "[$(timestamp)] [SKIP] Config not found: $cfg"
    continue
  fi

  if [[ "$SKIP_IF_ALREADY_IN_CSV" == "1" ]]; then
    if already_in_csv "$name"; then
      echo "[$(timestamp)] [SKIP] $name already in results/results.csv"
      continue
    fi
  fi

  ckpt_dir_raw="$(get_ckpt_dir_from_config "$cfg")"
  if [[ -z "$ckpt_dir_raw" ]]; then
    echo "[$(timestamp)] [FAIL] $name: train.ckpt_dir missing in $cfg"
    FAILED=$((FAILED+1))
    continue
  fi

  ckpt_path="$(find_best_ckpt "$cfg" "$ckpt_dir_raw" || true)"
  if [[ -z "$ckpt_path" ]]; then
    echo "[$(timestamp)] [FAIL] $name: best.pt not found."
    echo "         ckpt_dir_raw: $ckpt_dir_raw"
    echo "         hint: check ckpt folders under $ROOT_DIR/ckpts/"
    FAILED=$((FAILED+1))
    continue
  fi

  log_path="logs/eval_${name}.log"
  echo "[$(timestamp)] [RUN ] $name"
  echo "         cfg:  $cfg"
  echo "        ckpt:  $ckpt_path"
  echo "         log:  $log_path"

  python3 -m scripts.eval \
    --config "$cfg" \
    --ckpt "$ckpt_path" \
    --split test \
    --out_csv results/results.csv \
    --out_json_dir results/json \
    2>&1 | tee "$log_path"

  status=${PIPESTATUS[0]}
  if [[ $status -ne 0 ]]; then
    echo "[$(timestamp)] [FAIL] $name (exit=$status)"
    FAILED=$((FAILED+1))
  else
    echo "[$(timestamp)] [OK  ] $name"
  fi
  echo
done

echo "[$(timestamp)] Finished evaluations."
echo "  total configs considered: $TOTAL"
echo "  failed: $FAILED"
echo "  results csv: $ROOT_DIR/results/results.csv"
exit 0
