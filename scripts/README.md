# SLEIPNIR runtime scripts

This directory contains the three processing stages used by
`../run_inference.sh`.

## Install the existing scripts

From `/home/ubuntu/SLEIPNIR-V1.0.0`, run:

```bash
cp /home/ubuntu/csi-vial-fill-dataset/csi-training/crop.py \
  scripts/crop.py

cp /home/ubuntu/csi-vial-fill-dataset/csi-training/predict-vial-5class.py \
  scripts/predict-vial-5class.py
```

`apply_sleipnir_policy.py` is already supplied by the starter bundle.

## Script responsibilities

### `crop.py`

Reads full source images and creates 30 tight plus 30 context crops per complete
fixture-G frame. It also writes `index.csv`, which connects crop filenames to
source frames and fixture positions.

### `predict-vial-5class.py`

Loads the Round 3 checkpoint, preprocesses tight crops, and writes raw five-
class probabilities and predictions. Its raw frame decisions are retained for
traceability but are not the final SLEIPNIR operational output.

### `apply_sleipnir_policy.py`

Converts raw predictions into `REJECT`, `REVIEW`, `PASS`, or `NO_VIAL`, groups
repeated alerts into events, and produces the files under `results/`.

## Recommended run command

Run all three stages from the deployment root:

```bash
SLEIPNIR_PYTHON=/home/ubuntu/csi-vial-fill-dataset/csi-training/.venv/bin/python \
  ./run_inference.sh /path/to/images /path/to/new-output-directory
```

## Running each stage manually

The wrapper is preferred, but individual stages can be run for troubleshooting.

Create crops:

```bash
env -u LD_LIBRARY_PATH \
  /home/ubuntu/csi-vial-fill-dataset/csi-training/.venv/bin/python \
  scripts/crop.py /path/to/images \
  --positions config/twin_positions_px.csv \
  --fixture G \
  --out /path/to/run/crops
```

Run raw inference:

```bash
env -u LD_LIBRARY_PATH \
  /home/ubuntu/csi-vial-fill-dataset/csi-training/.venv/bin/python \
  scripts/predict-vial-5class.py \
  --checkpoint model/sleipnir-round3-v1.0.0.pt \
  --tight-dir /path/to/run/crops/tight \
  --manifest /path/to/run/crops/index.csv \
  --output-dir /path/to/run/raw_predictions \
  --batch-size 128 \
  --workers 4
```

Apply the SLEIPNIR policy:

```bash
env -u LD_LIBRARY_PATH \
  /home/ubuntu/csi-vial-fill-dataset/csi-training/.venv/bin/python \
  scripts/apply_sleipnir_policy.py \
  --predictions /path/to/run/raw_predictions/crop_predictions.csv \
  --policy config/inference_policy.json \
  --output-dir /path/to/run/results
```

Successful policy output includes position-action counts, frame-decision
counts, independent-event counts, and a final results directory. Use
`results/frame_summary.csv` and `results/event_summary.csv` for operations and
`results/review_queue.csv` for human review.