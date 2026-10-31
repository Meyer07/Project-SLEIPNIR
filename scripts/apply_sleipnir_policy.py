#!/usr/bin/env python3
"""Apply SLEIPNIR operating rules to raw five-class crop predictions."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


OUTPUT_FILES = (
    "operational_crop_predictions.csv",
    "frame_summary.csv",
    "event_summary.csv",
    "review_queue.csv",
    "run_summary.json",
)
TIMESTAMP_RE = re.compile(r"cap_seal_(\d{8}_\d{6}_\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("config/inference_policy.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing SLEIPNIR result files",
    )
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def load_policy(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Policy not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        policy = json.load(handle)

    required = {
        "project_name",
        "version",
        "expected_positions",
        "class_order",
        "automatic_reject_classes",
        "review_classes",
        "review_confidence_below",
        "review_top2_margin_below",
        "episode_gap_minutes",
    }
    missing = sorted(required - set(policy))
    if missing:
        raise ValueError(f"Policy is missing fields: {missing}")
    if policy["project_name"] != "SLEIPNIR":
        raise ValueError("Policy project_name must be SLEIPNIR")

    classes = set(policy["class_order"])
    configured = set(policy["automatic_reject_classes"]) | set(
        policy["review_classes"]
    )
    unknown = sorted(configured - classes)
    if unknown:
        raise ValueError(f"Policy contains unknown classes: {unknown}")
    return policy


def prepare_output(directory: Path, force: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    existing = [directory / name for name in OUTPUT_FILES]
    existing = [path for path in existing if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Output files already exist: {names}. Use a new directory or --force."
        )


def load_predictions(path: Path, policy: dict[str, object]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Predictions not found: {path}")
    frame = pd.read_csv(path, keep_default_na=False)
    classes = list(policy["class_order"])
    required = {
        "crop",
        "image",
        "fixture",
        "position",
        "prediction",
        "confidence",
        "top2_margin",
        "p_any_defect",
        *{f"p_{name}" for name in classes},
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {missing}")
    if frame["crop"].duplicated().any():
        example = frame.loc[frame["crop"].duplicated(), "crop"].iloc[0]
        raise ValueError(f"Duplicate crop prediction: {example}")
    invalid = sorted(set(frame["prediction"]) - set(classes))
    if invalid:
        raise ValueError(f"Unexpected predictions: {invalid}")
    frame["position"] = pd.to_numeric(frame["position"], errors="raise").astype(int)
    return frame


def apply_policy(frame: pd.DataFrame, policy: dict[str, object]) -> pd.DataFrame:
    reject_classes = set(policy["automatic_reject_classes"])
    review_classes = set(policy["review_classes"])
    confidence_limit = float(policy["review_confidence_below"])
    margin_limit = float(policy["review_top2_margin_below"])

    actions: list[str] = []
    reasons: list[str] = []
    for row in frame.itertuples(index=False):
        row_reasons: list[str] = []
        if row.prediction in reject_classes:
            action = "REJECT"
            row_reasons.append(f"automatic_reject:{row.prediction}")
        elif row.prediction in review_classes:
            action = "REVIEW"
            row_reasons.append(f"class_requires_review:{row.prediction}")
        elif row.confidence < confidence_limit:
            action = "REVIEW"
            row_reasons.append("low_confidence")
        elif row.top2_margin < margin_limit:
            action = "REVIEW"
            row_reasons.append("small_top2_margin")
        elif row.prediction == "pass":
            action = "PASS"
        elif row.prediction == "no_vial":
            action = "NO_VIAL"
        else:
            raise ValueError(f"No operating rule for class {row.prediction!r}")
        actions.append(action)
        reasons.append(";".join(row_reasons))

    result = frame.copy()
    result["operational_action"] = actions
    result["operational_reason"] = reasons
    return result


def summarize_frames(
    frame: pd.DataFrame,
    expected_positions: int,
) -> pd.DataFrame:
    expected = set(range(1, expected_positions + 1))
    rows: list[dict[str, object]] = []
    for (image, fixture), group in frame.groupby(["image", "fixture"], sort=True):
        group = group.sort_values("position")
        present = set(group["position"].astype(int))
        missing = sorted(expected - present)
        actions = set(group["operational_action"])
        if "REJECT" in actions:
            decision = "REJECT"
        elif "REVIEW" in actions:
            decision = "REVIEW"
        elif "PASS" in actions:
            decision = "PASS"
        else:
            decision = "NO_VIAL"

        def positions_for(action: str) -> str:
            selected = group[group["operational_action"].eq(action)]
            return ";".join(
                f"{int(row.position)}:{row.prediction}:{row.confidence:.3f}"
                for row in selected.itertuples(index=False)
            )

        rows.append(
            {
                "image": image,
                "fixture": fixture,
                "frame_decision": decision,
                "position_count": len(group),
                "complete_frame": not missing,
                "missing_positions": ";".join(str(value) for value in missing),
                "reject_count": int(group["operational_action"].eq("REJECT").sum()),
                "review_count": int(group["operational_action"].eq("REVIEW").sum()),
                "pass_count": int(group["operational_action"].eq("PASS").sum()),
                "no_vial_count": int(group["operational_action"].eq("NO_VIAL").sum()),
                "reject_positions": positions_for("REJECT"),
                "review_positions": positions_for("REVIEW"),
            }
        )
    return pd.DataFrame(rows)


def summarize_events(
    frame: pd.DataFrame,
    gap_minutes: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = frame[
        frame["operational_action"].isin(["REJECT", "REVIEW"])
    ].copy()
    if events.empty:
        empty = pd.DataFrame(
            columns=[
                "event_id",
                "operational_action",
                "prediction",
                "fixture",
                "position",
                "crops",
                "source_frames",
                "start",
                "end",
                "minimum_confidence",
                "mean_confidence",
                "maximum_confidence",
            ]
        )
        frame = frame.copy()
        frame["event_id"] = ""
        return empty, frame

    extracted = events["image"].str.extract(TIMESTAMP_RE, expand=False)
    events["timestamp"] = pd.to_datetime(
        extracted,
        format="%Y%m%d_%H%M%S_%f",
        errors="coerce",
    )
    unparsed = int(events["timestamp"].isna().sum())
    if unparsed:
        print(
            f"Warning: {unparsed} alert crop(s) have unparseable timestamps; "
            "each will start a separate episode."
        )

    group_columns = [
        "operational_action",
        "prediction",
        "fixture",
        "position",
    ]
    events = events.sort_values(group_columns + ["timestamp", "image"])
    gap = events.groupby(group_columns)["timestamp"].diff()
    new_episode = gap.isna() | gap.gt(pd.Timedelta(minutes=gap_minutes))
    events["episode_number"] = (
        new_episode.groupby([events[column] for column in group_columns])
        .cumsum()
        .astype(int)
    )
    events["event_id"] = (
        events["operational_action"]
        + "_"
        + events["prediction"]
        + "_"
        + events["fixture"].astype(str)
        + events["position"].astype(int).astype(str).str.zfill(2)
        + "_E"
        + events["episode_number"].astype(str).str.zfill(2)
    )

    summary = (
        events.groupby(
            [
                "event_id",
                "operational_action",
                "prediction",
                "fixture",
                "position",
            ]
        )
        .agg(
            crops=("crop", "size"),
            source_frames=("image", "nunique"),
            start=("timestamp", "min"),
            end=("timestamp", "max"),
            minimum_confidence=("confidence", "min"),
            mean_confidence=("confidence", "mean"),
            maximum_confidence=("confidence", "max"),
            minimum_p_any_defect=("p_any_defect", "min"),
            mean_p_any_defect=("p_any_defect", "mean"),
            maximum_p_any_defect=("p_any_defect", "max"),
        )
        .reset_index()
        .sort_values(["operational_action", "start", "position"])
    )

    event_map = events[["crop", "event_id"]]
    result = frame.merge(event_map, on="crop", how="left", validate="one_to_one")
    result["event_id"] = result["event_id"].fillna("")
    return summary, result


def main() -> None:
    args = parse_args()
    args.predictions = args.predictions.resolve()
    args.policy = args.policy.resolve()
    args.output_dir = args.output_dir.resolve()
    policy = load_policy(args.policy)
    prepare_output(args.output_dir, args.force)

    predictions = load_predictions(args.predictions, policy)
    operational = apply_policy(predictions, policy)
    event_summary, operational = summarize_events(
        operational,
        float(policy["episode_gap_minutes"]),
    )
    frame_summary = summarize_frames(
        operational,
        int(policy["expected_positions"]),
    )
    review_queue = operational[
        operational["operational_action"].eq("REVIEW")
    ].sort_values(["event_id", "image", "position"])

    atomic_csv(
        operational,
        args.output_dir / "operational_crop_predictions.csv",
    )
    atomic_csv(frame_summary, args.output_dir / "frame_summary.csv")
    atomic_csv(event_summary, args.output_dir / "event_summary.csv")
    atomic_csv(review_queue, args.output_dir / "review_queue.csv")

    action_counts = operational["operational_action"].value_counts()
    decision_counts = frame_summary["frame_decision"].value_counts()
    event_counts = event_summary["operational_action"].value_counts()
    run_summary = {
        "project_name": policy["project_name"],
        "version": policy["version"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": str(args.policy),
        "raw_predictions": str(args.predictions),
        "crops": len(operational),
        "frames": len(frame_summary),
        "complete_frames": int(frame_summary["complete_frame"].sum()),
        "incomplete_frames": int((~frame_summary["complete_frame"]).sum()),
        "position_actions": {
            name: int(action_counts.get(name, 0))
            for name in ("REJECT", "REVIEW", "PASS", "NO_VIAL")
        },
        "frame_decisions": {
            name: int(decision_counts.get(name, 0))
            for name in ("REJECT", "REVIEW", "PASS", "NO_VIAL")
        },
        "events": {
            name: int(event_counts.get(name, 0))
            for name in ("REJECT", "REVIEW")
        },
    }
    atomic_json(run_summary, args.output_dir / "run_summary.json")

    print("SLEIPNIR operational results")
    print("Position actions:")
    print(operational["operational_action"].value_counts().to_string())
    print("\nFrame decisions:")
    print(frame_summary["frame_decision"].value_counts().to_string())
    print("\nIndependent events:")
    if event_summary.empty:
        print("none")
    else:
        print(
            event_summary.groupby(["operational_action", "prediction"])
            .size()
            .rename("events")
            .to_string()
        )
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()

