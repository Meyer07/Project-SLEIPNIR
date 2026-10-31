#!/usr/bin/env python3
"""Verify the structure, configuration, scripts, coordinates, and checkpoint."""

from __future__ import annotations

import argparse
import json
import py_compile
from pathlib import Path

import pandas as pd


CLASSES = (
    "no_vial",
    "pass",
    "missing_cap",
    "crooked_cap",
    "damaged_vial",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--skip-checkpoint-load",
        action="store_true",
        help="Check that the model exists without loading it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    policy_path = root / "config/inference_policy.json"

    required = [
        root / "VERSION",
        root / "README.md",
        root / "MODEL_CARD.md",
        root / "requirements.txt",
        root / "run_inference.sh",
        root / "make_checksums.sh",
        root / "scripts/crop.py",
        root / "scripts/predict-vial-5class.py",
        root / "scripts/apply_sleipnir_policy.py",
        root / "config/twin_positions_px.csv",
        policy_path,
    ]
    missing = [
        str(path.relative_to(root))
        for path in required
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing deployment files: " + ", ".join(missing)
        )

    with policy_path.open("r", encoding="utf-8") as handle:
        policy = json.load(handle)

    if policy.get("project_name") != "SLEIPNIR":
        raise ValueError("Policy project_name is not SLEIPNIR")

    if tuple(policy.get("class_order", ())) != CLASSES:
        raise ValueError(
            "Policy class order does not match the model class order"
        )

    if int(policy.get("expected_positions", 0)) != 30:
        raise ValueError("Policy must expect 30 fixture positions")

    model_path = root / str(policy["model_checkpoint"])
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint is missing: {model_path}"
        )

    positions_path = root / "config/twin_positions_px.csv"
    positions = pd.read_csv(positions_path)

    required_position_columns = {"position", "u_px", "v_px"}
    missing_columns = sorted(
        required_position_columns - set(positions.columns)
    )
    if missing_columns:
        raise ValueError(
            f"Coordinate CSV is missing columns: {missing_columns}"
        )

    if "fixture" in positions.columns:
        positions = positions[
            positions["fixture"].astype(str).eq("G")
        ]

    actual_positions = set(
        pd.to_numeric(
            positions["position"],
            errors="raise",
        ).astype(int)
    )
    expected_positions = set(range(1, 31))

    if actual_positions != expected_positions:
        raise ValueError(
            "Fixture G positions differ from 1..30: "
            f"missing={sorted(expected_positions - actual_positions)}, "
            f"extra={sorted(actual_positions - expected_positions)}"
        )

    for relative in (
        "scripts/crop.py",
        "scripts/predict-vial-5class.py",
        "scripts/apply_sleipnir_policy.py",
    ):
        py_compile.compile(
            str(root / relative),
            doraise=True,
        )

    if not args.skip_checkpoint_load:
        import torch

        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
        )

        if tuple(checkpoint.get("classes", ())) != CLASSES:
            raise ValueError("Checkpoint class order is incorrect")

        if checkpoint.get("view") != "tight":
            raise ValueError(
                "Checkpoint is not a tight-crop model"
            )

        if "model_state_dict" not in checkpoint:
            raise ValueError(
                "Checkpoint is missing model_state_dict"
            )

        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("SLEIPNIR bundle verification passed")
    print(f"Root: {root}")
    print(f"Version: {policy['version']}")
    print(f"Model: {model_path.relative_to(root)}")
    print("Fixture G positions: 30")


if __name__ == "__main__":
    main()