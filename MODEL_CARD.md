# SLEIPNIR v1.0.0 Model Card

## Overview

SLEIPNIR is a five-class visual inspection system for fixture-G vial images.
The active model is the Round 3 tight-crop checkpoint. It predicts one class
for each of 30 fixture positions:

1. `no_vial`
2. `pass`
3. `missing_cap`
4. `crooked_cap`
5. `damaged_vial`

The deployment system separates the raw model prediction from the operational
action. Missing-cap and damaged-vial predictions automatically reject;
crooked-cap predictions require human review.

## Model details

| Item | Value |
| --- | --- |
| Product name | SLEIPNIR |
| Release | 1.0.0 |
| Active training round | Round 3 |
| Architecture | Pretrained ResNet18 with a five-class output layer |
| Automatic input | 72-pixel tight crop resized to 224 × 224 |
| Context crop | Human review only |
| Training rows | 5,746 usable labeled crops |
| Source frames | 200 |
| Capture sessions | 7 |
| Final training epochs | 5 |

### Round 3 training labels

| Class | Rows |
| --- | ---: |
| `no_vial` | 4,018 |
| `pass` | 1,199 |
| `missing_cap` | 119 |
| `crooked_cap` | 257 |
| `damaged_vial` | 153 |

Blank and `uncertain` rows were excluded from training.

## Held-out validation

Validation held out complete capture sessions so adjacent frames did not cross
between training and validation. The primary reported estimate uses Round 3
folds 0–3: 2,327 held-out crop predictions containing 87 defects.

| Metric | Result |
| --- | ---: |
| Balanced accuracy | 80.7% |
| Macro F1 | 78.7% |
| Any-defect recall | 71.3% (62/87) |
| Any-defect precision | 80.5% (62/77) |
| Any-defect F1 | 75.6% |
| False alarms per 100 nondefects | 0.67 |

### Exact per-class recall

| Class | Recall | Held-out support |
| --- | ---: | ---: |
| `no_vial` | 99.6% | 1,906 |
| `pass` | 97.3% | 334 |
| `missing_cap` | 90.0% | 60 |
| `crooked_cap` | 16.7% | 24 |
| `damaged_vial` | 100.0% | 3 |

The damaged-vial estimate is not reliable because only three independent
held-out examples were available. Crooked-cap recall is the primary weakness.

## Operational policy estimate

When only missing-cap and damaged-vial predictions automatically reject and
crooked-cap predictions go to review, the Round 3 held-out results indicate
approximately:

| Metric | Result |
| --- | ---: |
| Automatic defect precision | 95.1% |
| Automatic any-defect recall | 66.7% |
| False alarms per 100 nondefects | 0.13 |

Human review may recover some crooked caps that are not automatically rejected.
These figures are crop-level estimates, not guaranteed frame- or production-
level rates.

## Post-training audit evidence

Model-guided review found three sustained missing-cap episodes and four
sustained damaged-vial episodes. Blinded spot checks confirmed all sampled
start, middle, and end crops for those events. A subsequent re-audit found 58
rows originally labeled `pass` that reviewers consistently changed to
`damaged_vial`; the active Round 3 final model classified 56 of those 58 as
damaged.

This evidence supports the presence of useful damage signal, but it is post-hoc
and must not replace held-out validation metrics. Corrected labels were retained
in the Round 4 training archive; the Round 4 raw model was not promoted because
its false-alarm rate and macro F1 were worse overall.

## Intended use

SLEIPNIR is intended to assist fixture-G vial inspection under the camera,
lighting, cropping geometry, and product conditions represented by the pilot
dataset. Its outputs should feed an operational reject/review workflow with
traceable images and probabilities.

## Limitations

- Crooked-cap behavior varied substantially between capture sessions.
- Damaged-vial held-out support was extremely small.
- Many labeled crops are repeated frames of a small number of physical events;
  crop count is not equivalent to independent defect diversity.
- The system has not been validated for other fixtures, cameras, lenses,
  lighting, vial types, crop coordinates, or image naming conventions.
- Frame-level alert rates can be much higher than crop-level false-alarm rates
  because every frame contains 30 positions.
- Low-confidence and crooked-cap predictions require human review.
- The package is not a substitute for formal manufacturing validation,
  qualification, change control, or applicable regulatory requirements.

## Change control

Changing any of the following requires a new SLEIPNIR version and validation:

- Model checkpoint or class order
- Crop coordinates or crop dimensions
- Camera/lighting/fixture configuration
- Image preprocessing
- Automatic reject or review policy
- Confidence or episode-gap thresholds
- PyTorch/Torchvision behavior that changes inference output

The `checksums.sha256` file should be regenerated for every approved release.

