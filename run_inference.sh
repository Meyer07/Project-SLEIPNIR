#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 INPUT_IMAGE_DIRECTORY OUTPUT_DIRECTORY [BATCH_SIZE] [WORKERS]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$1"
OUTPUT_DIR="$2"
BATCH_SIZE="${3:-128}"
WORKERS="${4:-4}"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Input image directory not found: $INPUT_DIR" >&2
  exit 1
fi

if [[ -n "${SLEIPNIR_PYTHON:-}" ]]; then
  PYTHON_BIN="$SLEIPNIR_PYTHON"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

MODEL_PATH="$ROOT_DIR/model/sleipnir-round3-v1.0.0.pt"
POSITIONS_PATH="$ROOT_DIR/config/twin_positions_px.csv"
POLICY_PATH="$ROOT_DIR/config/inference_policy.json"
CROP_SCRIPT="$ROOT_DIR/scripts/crop.py"
PREDICT_SCRIPT="$ROOT_DIR/scripts/predict-vial-5class.py"
POLICY_SCRIPT="$ROOT_DIR/scripts/apply_sleipnir_policy.py"

for required in \
  "$MODEL_PATH" \
  "$POSITIONS_PATH" \
  "$POLICY_PATH" \
  "$CROP_SCRIPT" \
  "$PREDICT_SCRIPT" \
  "$POLICY_SCRIPT"; do
  if [[ ! -f "$required" ]]; then
    echo "Required deployment file is missing: $required" >&2
    exit 1
  fi
done

if [[ -e "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Output directory is not empty: $OUTPUT_DIR" >&2
  echo "Choose a new directory so an earlier run is not overwritten." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
CROP_DIR="$OUTPUT_DIR/crops"
RAW_DIR="$OUTPUT_DIR/raw_predictions"
RESULT_DIR="$OUTPUT_DIR/results"

echo "SLEIPNIR v1.0.0"
echo "Input:   $INPUT_DIR"
echo "Output:  $OUTPUT_DIR"
echo "Python:  $PYTHON_BIN"

echo
echo "[1/3] Creating fixture crops"
env -u LD_LIBRARY_PATH "$PYTHON_BIN" "$CROP_SCRIPT" \
  "$INPUT_DIR" \
  --positions "$POSITIONS_PATH" \
  --fixture G \
  --out "$CROP_DIR"

echo
echo "[2/3] Running the Round 3 five-class model"
env -u LD_LIBRARY_PATH "$PYTHON_BIN" "$PREDICT_SCRIPT" \
  --checkpoint "$MODEL_PATH" \
  --tight-dir "$CROP_DIR/tight" \
  --manifest "$CROP_DIR/index.csv" \
  --output-dir "$RAW_DIR" \
  --batch-size "$BATCH_SIZE" \
  --workers "$WORKERS"

echo
echo "[3/3] Applying SLEIPNIR policy and grouping events"
env -u LD_LIBRARY_PATH "$PYTHON_BIN" "$POLICY_SCRIPT" \
  --predictions "$RAW_DIR/crop_predictions.csv" \
  --policy "$POLICY_PATH" \
  --output-dir "$RESULT_DIR"

echo
echo "SLEIPNIR run complete"
echo "Results: $RESULT_DIR"
echo "Summary: $RESULT_DIR/run_summary.json"

