"""Command-line entry point for reproducible AeroGuard workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from aeroguard.analysis.statistics import build_day1_reports
from aeroguard.config import FIGURES_DIR, PROCESSED_DIR, RAW_DIR, REPORTS_DIR, ensure_directories
from aeroguard.data.download import download_dataset
from aeroguard.data.ingest import load_fd001
from aeroguard.data.rul import add_test_rul, add_train_rul
from aeroguard.data.validate import raise_for_invalid, validate_fd001_bundle
from aeroguard.ml.train import train_day2_models


def run_day1(*, force_download: bool = False) -> dict[str, object]:
    """Run download, validation, labeling, statistics, and visualization."""

    ensure_directories()
    print("[1/5] Downloading or reusing the official NASA archive...")
    manifest = download_dataset(force=force_download)

    print("[2/5] Parsing FD001 train, test, and RUL truth files...")
    train, test, truth = load_fd001()

    print("[3/5] Validating schema and engine-cycle business rules...")
    train_report, test_report, truth_issues = validate_fd001_bundle(train, test, truth)
    raise_for_invalid(train_report, test_report, truth_issues)

    print("[4/5] Constructing RUL and 30-cycle failure-risk targets...")
    train_labeled = add_train_rul(train)
    test_labeled = add_test_rul(test, truth)
    train_path = PROCESSED_DIR / "train_fd001_labeled.csv"
    test_path = PROCESSED_DIR / "test_fd001_labeled.csv"
    train_labeled.to_csv(train_path, index=False)
    test_labeled.to_csv(test_path, index=False)

    print("[5/5] Creating statistical reports and visualizations...")
    summary = build_day1_reports(train_labeled)
    result = {
        "source_sha256": manifest["archive_sha256"],
        "train_validation": train_report.to_dict(),
        "test_validation": test_report.to_dict(),
        "test_truth_valid": not truth_issues,
        "test_rows": int(len(test_labeled)),
        **summary,
    }
    print(json.dumps(result, indent=2))
    print(f"Day 1 complete. Figures: {FIGURES_DIR}")
    return result


def run_day2() -> dict[str, object]:
    """Run grouped validation, model comparison, tracking, and final evaluation."""

    ensure_directories()
    train_path = PROCESSED_DIR / "train_fd001_labeled.csv"
    test_path = PROCESSED_DIR / "test_fd001_labeled.csv"
    if not train_path.exists() or not test_path.exists():
        print("Day 1 labeled data is missing, so it will be generated first.")
        run_day1()

    print("[1/5] Loading the validated, labeled FD001 datasets...")
    train_labeled = pd.read_csv(train_path)
    test_labeled = pd.read_csv(test_path)
    print("[2/5] Splitting complete engines into training and validation groups...")
    print("[3/5] Training and tracking baseline, Ridge, Random Forest, and XGBoost...")
    print("[4/5] Selecting the champion using validation RMSE only...")
    result = train_day2_models(train_labeled, test_labeled)
    print("[5/5] Evaluating once on official test endpoints and saving artifacts...")
    print(json.dumps(result, indent=2))
    print("Day 2 complete. The official test set was not used to select the model.")
    return result


def run_day3() -> dict[str, object]:
    """Run LSTM sequence modeling and SHAP tree explanations."""

    from aeroguard.config import MODELS_DIR
    from aeroguard.ml.day3 import train_day3_models

    ensure_directories()
    train_path = PROCESSED_DIR / "train_fd001_labeled.csv"
    test_path = PROCESSED_DIR / "test_fd001_labeled.csv"
    required_day2 = (
        MODELS_DIR / "champion_metadata.json",
        MODELS_DIR / "champion_rul_model.joblib",
        MODELS_DIR / "candidate_random_forest.joblib",
    )
    if (
        not train_path.exists()
        or not test_path.exists()
        or any(not path.exists() for path in required_day2)
    ):
        print("Day 2 artifacts are missing, so Day 2 will be generated first.")
        run_day2()

    print("[1/6] Loading labeled data and reproducing the engine-group split...")
    train_labeled = pd.read_csv(train_path)
    test_labeled = pd.read_csv(test_path)
    print("[2/6] Scaling training features and creating 30-cycle sequences...")
    print("[3/6] Training the LSTM with validation-based early stopping...")
    print("[4/6] Comparing LSTM and Random Forest on identical endpoints...")
    print("[5/6] Refitting the LSTM and evaluating official test endpoints...")
    result = train_day3_models(train_labeled, test_labeled)
    print("[6/6] Calculating global and local SHAP explanations...")
    print(json.dumps(result, indent=2))
    print("Day 3 complete. Validation, not test data, selected the accuracy winner.")
    return result


def run_day4() -> dict[str, object]:
    """Build the local knowledge index and a grounded reliability brief."""

    from dotenv import load_dotenv

    from aeroguard.config import MODELS_DIR, PROJECT_ROOT
    from aeroguard.rag.day4 import run_day4_copilot

    ensure_directories()
    required_day3 = (
        REPORTS_DIR / "day3_summary.json",
        MODELS_DIR / "champion_rul_model.joblib",
        MODELS_DIR / "champion_metadata.json",
        MODELS_DIR / "lstm_rul_model.pt",
        MODELS_DIR / "lstm_feature_scaler.joblib",
        MODELS_DIR / "lstm_metadata.json",
    )
    if any(not path.exists() for path in required_day3):
        print("Day 3 artifacts are missing, so Day 3 will be generated first.")
        run_day3()

    load_dotenv(PROJECT_ROOT / ".env")
    print("[1/5] Loading and fingerprinting the reliability knowledge base...")
    print("[2/5] Building the local TF-IDF retrieval index...")
    print("[3/5] Evaluating retrieval with labeled questions...")
    print("[4/5] Building a real model-and-SHAP evidence packet...")
    result = run_day4_copilot()
    print("[5/5] Generating a cited reliability brief without changing the RUL...")
    print(json.dumps(result, indent=2))
    print("Day 4 complete. The copilot used model evidence and cited retrieved text.")
    return result


def run_day5() -> dict[str, object]:
    """Verify API readiness and create the final operations-console report."""

    from aeroguard.api.day5 import build_day5_summary

    ensure_directories()
    required_day4 = (
        REPORTS_DIR / "day4_summary.json",
        REPORTS_DIR / "day4_reliability_brief.json",
        REPORTS_DIR / "day3_test_predictions.csv",
    )
    if any(not path.exists() for path in required_day4):
        print("Day 4 artifacts are missing, so Day 4 will be generated first.")
        run_day4()

    print("[1/4] Verifying the Day 1-4 models, predictions, and RAG artifacts...")
    print("[2/4] Building the fleet-priority and model-monitoring snapshot...")
    result = build_day5_summary()
    print("[3/4] Confirming the real-data API and dashboard contract...")
    print("[4/4] Saving the operational-readiness report...")
    print(json.dumps(result, indent=2))
    print("Day 5 complete. Start the dashboard with run_day5.ps1 or make app.")
    return result


def clean_generated() -> None:
    """Delete only reproducible processed outputs, never the downloaded source."""

    targets = [
        PROCESSED_DIR / "train_fd001_labeled.csv",
        PROCESSED_DIR / "test_fd001_labeled.csv",
        REPORTS_DIR / "day1_summary.json",
        REPORTS_DIR / "engine_summary.csv",
        REPORTS_DIR / "sensor_summary.csv",
    ]
    targets.extend(FIGURES_DIR.glob("*.png"))
    for target in targets:
        Path(target).unlink(missing_ok=True)
    print("Generated Day 1 outputs removed; raw NASA source retained.")


def main() -> None:
    parser = argparse.ArgumentParser(description="AeroGuard AI workflows")
    subparsers = parser.add_subparsers(dest="command", required=True)
    download_parser = subparsers.add_parser("download", help="Download FD001 source")
    download_parser.add_argument("--force", action="store_true")
    day1_parser = subparsers.add_parser("day1", help="Run the complete Day 1 workflow")
    day1_parser.add_argument("--force-download", action="store_true")
    subparsers.add_parser("day2", help="Run the complete Day 2 ML workflow")
    subparsers.add_parser("day3", help="Run LSTM and SHAP explainability")
    subparsers.add_parser("day4", help="Run grounded reliability RAG")
    subparsers.add_parser("day5", help="Verify and prepare the operations console")
    subparsers.add_parser("clean-generated", help="Remove reproducible Day 1 outputs")
    arguments = parser.parse_args()

    if arguments.command == "download":
        print(json.dumps(download_dataset(force=arguments.force), indent=2))
    elif arguments.command == "day1":
        run_day1(force_download=arguments.force_download)
    elif arguments.command == "day2":
        run_day2()
    elif arguments.command == "day3":
        run_day3()
    elif arguments.command == "day4":
        run_day4()
    elif arguments.command == "day5":
        run_day5()
    elif arguments.command == "clean-generated":
        clean_generated()


if __name__ == "__main__":
    main()
