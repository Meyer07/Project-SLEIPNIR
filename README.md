# SLEIPNIR v1.0.0

SLEIPNIR is the packaged Vial Line inspection system. It crops 30 fixture-G
positions from each source image, runs the selected Round 3 tight-crop model,
applies the validated operating policy, and groups repeated detections into
events.

## Quick installation on the current Brev instance

The commands below assume the training project remains at
`/home/ubuntu/csi-vial-fill-dataset/csi-training` and the deployment directory
is `/home/ubuntu/SLEIPNIR-V1.0.0`.

Extract and enter the starter bundle:

```bash
cd /home/ubuntu
tar -xzf SLEIPNIR-v1.0.0-starter.tar.gz
mv SLEIPNIR-v1.0.0-starter SLEIPNIR-V1.0.0
cd SLEIPNIR-V1.0.0
```

Copy the four existing runtime assets:

```bash
mkdir -p model config scripts

cp /home/ubuntu/csi-vial-fill-dataset/csi-training/vial_round3_final_tight/final_model.pt \
  model/sleipnir-round3-v1.0.0.pt

cp /home/ubuntu/csi-vial-fill-dataset/csi-training/twin_positions_px.csv \
  config/twin_positions_px.csv

cp /home/ubuntu/csi-vial-fill-dataset/csi-training/crop.py \
  scripts/crop.py

cp /home/ubuntu/csi-vial-fill-dataset/csi-training/predict-vial-5class.py \
  scripts/predict-vial-5class.py
```

Make the launchers executable and verify the installation:

```bash
chmod +x run_inference.sh make_checksums.sh verify_bundle.py scripts/*.py

env -u LD_LIBRARY_PATH \
  /home/ubuntu/csi-vial-fill-dataset/csi-training/.venv/bin/python \
  verify_bundle.py
```

Successful verification ends with:

```text
SLEIPNIR bundle verification passed
Version: 1.0.0
Model: model/sleipnir-round3-v1.0.0.pt
Fixture G positions: 30
```

Run a batch of images:

```bash
SLEIPNIR_PYTHON=/home/ubuntu/csi-vial-fill-dataset/csi-training/.venv/bin/python \
  ./run_inference.sh \
  /home/ubuntu/csi-vial-fill-dataset/cap_seal/images \
  /home/ubuntu/sleipnir-runs/run-001
```

Always use a new or empty output directory for each run.

## Files you must copy into this starter bundle

The starter already contains all newly written deployment files. Copy these
four existing project assets into the indicated destinations:

| Existing source file | Destination in this bundle |
| --- | --- |
| `vial_round3_final_tight/final_model.pt` | `model/sleipnir-round3-v1.0.0.pt` |
| `crop.py` | `scripts/crop.py` |
| `predict-vial-5class.py` | `scripts/predict-vial-5class.py` |
| `twin_positions_px.csv` | `config/twin_positions_px.csv` |

From the training project directory, an example copy sequence is:

```bash
cp vial_round3_final_tight/final_model.pt \
  /path/to/SLEIPNIR-v1.0.0/model/sleipnir-round3-v1.0.0.pt

cp crop.py \
  /path/to/SLEIPNIR-v1.0.0/scripts/crop.py

cp predict-vial-5class.py \
  /path/to/SLEIPNIR-v1.0.0/scripts/predict-vial-5class.py

cp twin_positions_px.csv \
  /path/to/SLEIPNIR-v1.0.0/config/twin_positions_px.csv
```

Do not substitute the Round 4 checkpoint. Round 3 remains the selected model
because its held-out macro F1, defect precision, and false-alarm rate were
better overall.

## Directory layout

```text
SLEIPNIR-v1.0.0/
  model/
    sleipnir-round3-v1.0.0.pt
  config/
    inference_policy.json
    twin_positions_px.csv
  scripts/
    crop.py
    predict-vial-5class.py
    apply_sleipnir_policy.py
  MODEL_CARD.md
  README.md
  VERSION
  requirements.txt
  verify_bundle.py
  run_inference.sh
  make_checksums.sh
```

## Operating policy

SLEIPNIR uses the model's tight-crop prediction as its primary signal:

| Model prediction | Operational action |
| --- | --- |
| `missing_cap` | `REJECT` |
| `damaged_vial` | `REJECT` |
| `crooked_cap` | `REVIEW` |
| Low-confidence or small-margin result | `REVIEW` |
| `pass` | `PASS` |
| `no_vial` | `NO_VIAL` |

Context crops are retained for human inspection but do not change the
automatic decision. Context/tight automatic fusion was tested and did not
improve held-out results.

Repeated `REJECT` and `REVIEW` predictions at the same fixture position are
grouped into one event when the gap between detections is no more than five
minutes. The event grouping reduces review workload; it does not suppress
per-frame machine decisions.

## Environment setup

The validated environment used Python 3.12, PyTorch 2.14.1+cu130,
Torchvision 0.29.1+cu130, and an NVIDIA L40S.

When deploying on the existing Brev environment, the current `.venv` may be
reused. For a new system, create a virtual environment and install the matching
CUDA-enabled PyTorch build plus the remaining requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate

# Install the approved CUDA 13.0 PyTorch/Torchvision wheels for the target.
pip install -r requirements.txt
```

The exact CUDA wheel source can vary by infrastructure. Do not silently replace
the CUDA build with an untested CPU or different-CUDA build.

## Complete and verify the bundle

After copying the four existing assets:

```bash
chmod +x run_inference.sh make_checksums.sh scripts/*.py verify_bundle.py

env -u LD_LIBRARY_PATH .venv/bin/python verify_bundle.py
```

The verifier confirms:

- Required files are present.
- The policy and checkpoint use the same five-class order.
- The checkpoint is a tight-crop model.
- Fixture G contains positions 1 through 30.
- All Python deployment scripts compile.
- PyTorch and CUDA can load the trusted checkpoint.

Only load a checkpoint produced and controlled by this project. PyTorch
checkpoint loading is not safe for arbitrary untrusted files.

## Run SLEIPNIR

Give the runner a directory of source images and a new output directory:

```bash
./run_inference.sh \
  /path/to/source/images \
  /path/to/sleipnir-run-001
```

Optional third and fourth arguments set batch size and worker count:

```bash
./run_inference.sh /path/to/images /path/to/run 128 4
```

To choose a specific Python interpreter:

```bash
SLEIPNIR_PYTHON=/path/to/python \
  ./run_inference.sh /path/to/images /path/to/run
```

The runner refuses to write into a nonempty output directory.

### Expected terminal output

The runner reports three stages:

```text
SLEIPNIR v1.0.0
[1/3] Creating fixture crops
[2/3] Running the Round 3 five-class model
[3/3] Applying SLEIPNIR policy and grouping events
SLEIPNIR run complete
```

The cropping stage should create 30 tight crops and 30 context crops per
complete source frame. The prediction stage reports progress in crop counts.
The final policy stage prints position-action, frame-decision, and independent-
event counts.

## Run outputs

Each run contains:

```text
run-directory/
  crops/
    tight/
    context/
    index.csv
  raw_predictions/
    crop_predictions.csv
    frame_summary.csv
    uncertain_review.csv
    run_summary.json
  results/
    operational_crop_predictions.csv
    frame_summary.csv
    event_summary.csv
    review_queue.csv
    run_summary.json
```

Use the files under `results/` for operational decisions. The files under
`raw_predictions/` preserve the model's unmodified output for traceability.

### What a correct result should look like

- `operational_crop_predictions.csv` has one row per fixture position. A run
  with 100 complete source frames should contain 3,000 rows.
- `frame_summary.csv` has one row per source frame and `position_count=30` for
  every complete frame.
- `frame_decision` is one of `REJECT`, `REVIEW`, `PASS`, or `NO_VIAL`.
- `event_summary.csv` contains fewer rows than the crop alert count when the
  same physical condition persists across consecutive frames.
- `review_queue.csv` contains every crooked-cap or low-confidence position that
  requires a person to inspect it.
- `run_summary.json` records the SLEIPNIR version and aggregate result counts.

Treat `REJECT` as an automatic machine action, `REVIEW` as a human-decision
queue, `PASS` as a frame containing acceptable vials and no higher-priority
condition, and `NO_VIAL` as a frame with no detected vial and no alert.

### Important result files

- `operational_crop_predictions.csv`: probabilities, model class, action,
  reason, and event ID for every fixture position.
- `frame_summary.csv`: one decision per source frame.
- `event_summary.csv`: repeated alerts collapsed into independent events.
- `review_queue.csv`: every position requiring human review.
- `run_summary.json`: policy version and aggregate counts.

## Integrity checks

After all files have been copied and verification has passed, generate hashes:

```bash
./make_checksums.sh
```

Verify a transferred bundle with:

```bash
sha256sum -c checksums.sha256
```

Regenerate checksums whenever any packaged file intentionally changes, and
increase the SLEIPNIR version when model or decision behavior changes.

## What is intentionally excluded

The deployment bundle does not require:

- Training source images or the 2 GB crop dataset
- Round 1, Round 2, or fold checkpoints
- Round 4 experimental checkpoint
- Training manifests or audit working files
- Training logs
- Context-model checkpoints

Keep those materials in the training archive, not in the runtime bundle.