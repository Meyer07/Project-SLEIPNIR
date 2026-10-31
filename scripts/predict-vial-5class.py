"""Run the final five-class Vial Line model on tight vial crops.

The script performs five deployment steps:

1. Load and validate the trusted ``final_model.pt`` checkpoint.
2. Find tight crop PNGs and recover their source-frame/position metadata.
3. Predict one of five classes and all five probabilities for every crop.
4. Group positions into source frames and produce a PASS/REJECT/NO_VIAL result.
5. Write detailed CSVs, a run summary, and a low-confidence review queue.

The automatic decision intentionally uses tight crops only. The separately
trained context model did not improve held-out gated-fusion performance.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_CLASSES = (
    "no_vial",
    "pass",
    "missing_cap",
    "crooked_cap",
    "damaged_vial",
)
DEFECT_CLASSES = ("missing_cap", "crooked_cap", "damaged_vial")
OUTPUT_FILENAMES = (
    "crop_predictions.csv",
    "frame_summary.csv",
    "uncertain_review.csv",
    "run_summary.json",
)
CROP_NAME_RE = re.compile(
    r"^(?P<image_stem>.+)__(?P<fixture>[A-Za-z]+)(?P<position>\d+)\.png$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("vial_round3_final_tight/final_model.pt"),
        help="Trusted checkpoint produced by train-vial-5class.py",
    )
    parser.add_argument(
        "--tight-dir",
        type=Path,
        default=Path("crops_all/tight"),
        help="Directory containing tight crop PNGs",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional CSV with crop, image, fixture, and position columns. "
            "Without it, metadata is parsed from crop filenames."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("vial_inference_results"),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="auto uses CUDA when available",
    )
    parser.add_argument(
        "--review-confidence",
        type=float,
        default=0.70,
        help="Review predictions whose largest class probability is below this value",
    )
    parser.add_argument(
        "--review-margin",
        type=float,
        default=0.15,
        help="Review predictions whose top-two probability margin is below this value",
    )
    parser.add_argument(
        "--expected-positions",
        type=int,
        default=30,
        help="Expected fixture positions per source frame",
    )
    parser.add_argument(
        "--require-complete-frames",
        action="store_true",
        help="Fail instead of marking frames that do not contain every expected position",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for PNGs below --tight-dir recursively",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing inference result files in --output-dir",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.tight_dir.is_dir():
        raise NotADirectoryError(f"Tight-crop directory not found: {args.tight_dir}")
    if args.manifest is not None and not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative")
    if args.expected_positions < 1:
        raise ValueError("--expected-positions must be at least 1")
    for name in ("review_confidence", "review_margin"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = [args.output_dir / name for name in OUTPUT_FILENAMES]
    existing = [path for path in existing if path.exists()]
    if existing and not args.force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Output file(s) already exist in {args.output_dir}: {names}. "
            "Use a new --output-dir or pass --force."
        )


def parse_crop_name(crop_name: str) -> dict[str, object]:
    match = CROP_NAME_RE.match(Path(crop_name).name)
    if match is None:
        raise ValueError(
            f"Cannot parse crop filename {crop_name!r}; expected a name like "
            "cap_seal_...__G01.png. Supply --manifest if filenames differ."
        )
    return {
        "image": f"{match.group('image_stem')}.jpg",
        "fixture": match.group("fixture").upper(),
        "position": int(match.group("position")),
    }


def discover_without_manifest(args: argparse.Namespace) -> pd.DataFrame:
    iterator = args.tight_dir.rglob("*.png") if args.recursive else args.tight_dir.glob("*.png")
    paths = sorted(path for path in iterator if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No PNG crops found in {args.tight_dir}")

    basenames = [path.name for path in paths]
    duplicate_names = pd.Series(basenames)[pd.Series(basenames).duplicated()].unique().tolist()
    if duplicate_names:
        raise ValueError(
            "Crop basenames must be unique; duplicate example: "
            f"{duplicate_names[0]}. Use one flat crop directory."
        )

    rows = []
    for path in paths:
        metadata = parse_crop_name(path.name)
        rows.append({"crop": path.name, "crop_path": str(path.resolve()), **metadata})
    return pd.DataFrame(rows)


def discover_with_manifest(args: argparse.Namespace) -> pd.DataFrame:
    manifest = pd.read_csv(args.manifest, keep_default_na=False)
    required = {"crop", "image", "fixture", "position"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    if manifest["crop"].astype(str).duplicated().any():
        example = manifest.loc[manifest["crop"].astype(str).duplicated(), "crop"].iloc[0]
        raise ValueError(f"Manifest contains duplicate crop name: {example}")

    rows = manifest[["crop", "image", "fixture", "position"]].copy()
    rows["crop"] = rows["crop"].astype(str)
    rows["image"] = rows["image"].astype(str)
    rows["fixture"] = rows["fixture"].astype(str).str.upper()
    rows["position"] = pd.to_numeric(rows["position"], errors="raise").astype(int)
    rows["crop_path"] = rows["crop"].map(
        lambda name: str((args.tight_dir / name).resolve())
    )
    exists = rows["crop_path"].map(lambda value: Path(value).is_file())
    missing_count = int((~exists).sum())
    if missing_count:
        print(
            f"Warning: skipping {missing_count} manifest row(s) whose tight crop is not "
            f"present in {args.tight_dir}"
        )
    rows = rows[exists].copy()
    if rows.empty:
        raise FileNotFoundError(
            f"None of the manifest crops were found in {args.tight_dir}"
        )
    return rows


def load_inputs(args: argparse.Namespace) -> pd.DataFrame:
    frame = (
        discover_with_manifest(args)
        if args.manifest is not None
        else discover_without_manifest(args)
    )
    if frame.duplicated(["image", "fixture", "position"]).any():
        duplicate = frame.loc[
            frame.duplicated(["image", "fixture", "position"]),
            ["image", "fixture", "position"],
        ].iloc[0]
        raise ValueError(
            "More than one crop maps to the same source-frame position: "
            f"{duplicate.to_dict()}"
        )
    valid_position = frame["position"].between(1, args.expected_positions)
    if not valid_position.all():
        example = frame.loc[~valid_position, ["crop", "position"]].iloc[0].to_dict()
        raise ValueError(
            f"Position outside 1..{args.expected_positions}: {example}"
        )
    return frame.sort_values(["image", "fixture", "position"]).reset_index(drop=True)


def load_model_and_metadata(args: argparse.Namespace):
    import torch
    from torch import nn
    from torchvision.models import resnet18

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    device_name = (
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    device = torch.device(device_name)

    # PyTorch checkpoints can execute code while loading. Only use a checkpoint
    # produced by this project and never an untrusted downloaded file.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain model_state_dict")
    classes = tuple(checkpoint.get("classes", ()))
    if classes != EXPECTED_CLASSES:
        raise ValueError(
            f"Checkpoint classes/order {classes} do not match {EXPECTED_CLASSES}"
        )
    architecture = checkpoint.get("architecture", "resnet18")
    if architecture != "resnet18":
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
    view = checkpoint.get("view", "")
    if view and view != "tight":
        raise ValueError(f"This script requires a tight-crop checkpoint, found view={view!r}")

    image_size = tuple(int(value) for value in checkpoint.get("image_size", (224, 224)))
    if len(image_size) != 2 or min(image_size) < 1:
        raise ValueError(f"Invalid checkpoint image_size: {image_size}")
    normalization = checkpoint.get("normalization", {})
    mean = tuple(float(value) for value in normalization.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(float(value) for value in normalization.get("std", (0.229, 0.224, 0.225)))
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("Checkpoint normalization must contain three-channel mean/std")

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    state_dict = checkpoint["model_state_dict"]
    if state_dict and all(str(key).startswith("module.") for key in state_dict):
        state_dict = {str(key)[7:]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    metadata = {
        "classes": classes,
        "architecture": architecture,
        "view": "tight",
        "image_size": image_size,
        "mean": mean,
        "std": std,
        "device": device,
        "trained_epochs": checkpoint.get("trained_epochs"),
        "training_rows": checkpoint.get("training_rows"),
        "training_frames": checkpoint.get("training_frames"),
        "training_sessions": checkpoint.get("training_sessions"),
    }
    return model, metadata


def score_crops(
    inputs: pd.DataFrame,
    model,
    metadata: dict[str, object],
    args: argparse.Namespace,
) -> pd.DataFrame:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize(metadata["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=metadata["mean"], std=metadata["std"]),
        ]
    )

    class InferenceDataset(Dataset):
        def __init__(self, frame: pd.DataFrame):
            self.frame = frame.reset_index(drop=True)

        def __len__(self) -> int:
            return len(self.frame)

        def __getitem__(self, index: int):
            path = self.frame.iloc[index]["crop_path"]
            try:
                with Image.open(path) as image:
                    tensor = transform(image.convert("RGB"))
            except Exception as exc:
                raise RuntimeError(f"Could not read crop {path}: {exc}") from exc
            return tensor, index

    device = metadata["device"]
    loader = DataLoader(
        InferenceDataset(inputs),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    probability_rows = np.zeros((len(inputs), len(EXPECTED_CLASSES)), dtype=np.float32)
    processed = 0
    with torch.inference_mode():
        for images, indices in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
            probability_rows[indices.numpy()] = probabilities
            processed += len(indices)
            if processed % 6400 < len(indices) or processed == len(inputs):
                print(f"  scored {processed}/{len(inputs)} crops")

    output = inputs[["crop", "image", "fixture", "position", "crop_path"]].copy()
    for index, class_name in enumerate(EXPECTED_CLASSES):
        output[f"p_{class_name}"] = probability_rows[:, index]
    top_indices = probability_rows.argmax(axis=1)
    sorted_probabilities = np.sort(probability_rows, axis=1)
    output["prediction"] = [EXPECTED_CLASSES[index] for index in top_indices]
    output["confidence"] = sorted_probabilities[:, -1]
    output["top2_margin"] = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    defect_indices = [EXPECTED_CLASSES.index(name) for name in DEFECT_CLASSES]
    output["p_any_defect"] = probability_rows[:, defect_indices].sum(axis=1)
    output["is_defect"] = output["prediction"].isin(DEFECT_CLASSES)

    review_reasons = []
    for row in output.itertuples(index=False):
        reasons = []
        if row.confidence < args.review_confidence:
            reasons.append("low_confidence")
        if row.top2_margin < args.review_margin:
            reasons.append("small_top2_margin")
        review_reasons.append(";".join(reasons))
    output["review_reason"] = review_reasons
    output["needs_review"] = output["review_reason"].ne("")
    return output


def summarize_frames(
    predictions: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    expected = set(range(1, args.expected_positions + 1))
    rows = []
    for (image, fixture), group in predictions.groupby(["image", "fixture"], sort=True):
        group = group.sort_values("position")
        present = set(int(value) for value in group["position"])
        missing = sorted(expected - present)
        defect_rows = group[group["is_defect"]]
        if not defect_rows.empty:
            decision = "REJECT"
        elif group["prediction"].eq("pass").any():
            decision = "PASS"
        else:
            decision = "NO_VIAL"

        defect_positions = ";".join(
            f"{int(row.position)}:{row.prediction}:{row.confidence:.3f}"
            for row in defect_rows.itertuples(index=False)
        )
        counts = group["prediction"].value_counts()
        rows.append(
            {
                "image": image,
                "fixture": fixture,
                "frame_decision": decision,
                "position_count": len(group),
                "complete_frame": not missing,
                "missing_positions": ";".join(str(value) for value in missing),
                **{
                    f"pred_{class_name}": int(counts.get(class_name, 0))
                    for class_name in EXPECTED_CLASSES
                },
                "defect_count": len(defect_rows),
                "defect_positions": defect_positions,
                "review_count": int(group["needs_review"].sum()),
                "max_p_any_defect": float(group["p_any_defect"].max()),
                "minimum_confidence": float(group["confidence"].min()),
            }
        )
    summary = pd.DataFrame(rows)
    incomplete = summary[~summary["complete_frame"]]
    if args.require_complete_frames and not incomplete.empty:
        example = incomplete.iloc[0][
            ["image", "fixture", "position_count", "missing_positions"]
        ].to_dict()
        raise ValueError(
            f"Found {len(incomplete)} incomplete frame(s); first example: {example}"
        )
    return summary


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_write_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def write_outputs(
    predictions: pd.DataFrame,
    frame_summary: pd.DataFrame,
    metadata: dict[str, object],
    args: argparse.Namespace,
) -> None:
    review = predictions[predictions["needs_review"]].copy()
    review = review.sort_values(
        ["confidence", "top2_margin", "image", "position"],
        ascending=[True, True, True, True],
    )
    crop_output = predictions.drop(columns=["crop_path"])
    review_output = review.drop(columns=["crop_path"])
    atomic_write_csv(crop_output, args.output_dir / "crop_predictions.csv")
    atomic_write_csv(frame_summary, args.output_dir / "frame_summary.csv")
    atomic_write_csv(review_output, args.output_dir / "uncertain_review.csv")

    class_counts = crop_output["prediction"].value_counts().reindex(
        EXPECTED_CLASSES, fill_value=0
    )
    decision_counts = frame_summary["frame_decision"].value_counts()
    run_summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "tight_dir": str(args.tight_dir),
        "manifest": str(args.manifest) if args.manifest is not None else None,
        "output_dir": str(args.output_dir),
        "architecture": metadata["architecture"],
        "classes": list(EXPECTED_CLASSES),
        "device": str(metadata["device"]),
        "image_size": list(metadata["image_size"]),
        "automatic_input_view": "tight_only",
        "crop_count": len(crop_output),
        "frame_count": len(frame_summary),
        "complete_frames": int(frame_summary["complete_frame"].sum()),
        "incomplete_frames": int((~frame_summary["complete_frame"]).sum()),
        "review_count": len(review_output),
        "review_confidence": args.review_confidence,
        "review_margin": args.review_margin,
        "prediction_counts": {
            name: int(class_counts[name]) for name in EXPECTED_CLASSES
        },
        "frame_decision_counts": {
            name: int(decision_counts.get(name, 0))
            for name in ("PASS", "REJECT", "NO_VIAL")
        },
        "checkpoint_training": {
            "epochs": metadata["trained_epochs"],
            "rows": metadata["training_rows"],
            "frames": metadata["training_frames"],
            "sessions": metadata["training_sessions"],
        },
        "performance_note": (
            "Do not estimate deployment accuracy from this inference run. Use the "
            "previous session-held-out validation metrics."
        ),
    }
    atomic_write_json(run_summary, args.output_dir / "run_summary.json")


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.tight_dir = args.tight_dir.resolve()
    args.manifest = args.manifest.resolve() if args.manifest is not None else None
    args.output_dir = args.output_dir.resolve()
    validate_args(args)

    print("Step 1/5: loading and validating checkpoint")
    model, metadata = load_model_and_metadata(args)
    print(f"  device: {metadata['device']}")
    print(f"  classes: {', '.join(metadata['classes'])}")

    print("Step 2/5: finding tight crops")
    inputs = load_inputs(args)
    print(
        f"  found {len(inputs)} crops from "
        f"{inputs[['image', 'fixture']].drop_duplicates().shape[0]} source frames"
    )

    print("Step 3/5: classifying crops")
    predictions = score_crops(inputs, model, metadata, args)

    print("Step 4/5: building frame decisions")
    frame_summary = summarize_frames(predictions, args)
    print(
        "  decisions: "
        + ", ".join(
            f"{name}={int((frame_summary['frame_decision'] == name).sum())}"
            for name in ("PASS", "REJECT", "NO_VIAL")
        )
    )

    print("Step 5/5: writing outputs and review queue")
    write_outputs(predictions, frame_summary, metadata, args)
    print(f"  crop predictions: {args.output_dir / 'crop_predictions.csv'}")
    print(f"  frame decisions:  {args.output_dir / 'frame_summary.csv'}")
    print(f"  review queue:     {args.output_dir / 'uncertain_review.csv'}")
    print(f"  run summary:      {args.output_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()

