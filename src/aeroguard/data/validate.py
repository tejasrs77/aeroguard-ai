"""Transparent schema and business-rule checks for C-MAPSS data."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from aeroguard.config import CMAPSS_COLUMNS


@dataclass(frozen=True)
class ValidationIssue:
    rule: str
    count: int
    detail: str


@dataclass(frozen=True)
class ValidationReport:
    dataset: str
    rows: int
    engines: int
    issues: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "rows": self.rows,
            "engines": self.engines,
            "valid": self.valid,
            "issues": [asdict(issue) for issue in self.issues],
        }


def validate_sensor_frame(frame: pd.DataFrame, dataset: str) -> ValidationReport:
    """Check rules that must hold before feature engineering or modeling."""

    issues: list[ValidationIssue] = []
    missing_columns = [column for column in CMAPSS_COLUMNS if column not in frame]
    unexpected_columns = [column for column in frame if column not in CMAPSS_COLUMNS]
    if missing_columns or unexpected_columns:
        issues.append(
            ValidationIssue(
                "schema_columns",
                len(missing_columns) + len(unexpected_columns),
                f"missing={missing_columns}; unexpected={unexpected_columns}",
            )
        )
        return ValidationReport(dataset, len(frame), 0, tuple(issues))

    missing_values = int(frame[CMAPSS_COLUMNS].isna().sum().sum())
    if missing_values:
        issues.append(
            ValidationIssue(
                "required_values", missing_values, "Every source value is required."
            )
        )

    numeric = frame[CMAPSS_COLUMNS].select_dtypes(include=[np.number])
    non_numeric_columns = sorted(set(CMAPSS_COLUMNS) - set(numeric.columns))
    if non_numeric_columns:
        issues.append(
            ValidationIssue(
                "numeric_types",
                len(non_numeric_columns),
                f"Non-numeric columns: {non_numeric_columns}",
            )
        )
    elif not np.isfinite(numeric.to_numpy(dtype=float)).all():
        non_finite = int((~np.isfinite(numeric.to_numpy(dtype=float))).sum())
        issues.append(
            ValidationIssue(
                "finite_values", non_finite, "NaN and infinite values are invalid."
            )
        )

    invalid_ids = int(((frame["engine_id"] <= 0) | (frame["cycle"] <= 0)).sum())
    if invalid_ids:
        issues.append(
            ValidationIssue(
                "positive_engine_and_cycle",
                invalid_ids,
                "Engine IDs and cycles must be positive.",
            )
        )

    duplicate_keys = int(frame.duplicated(["engine_id", "cycle"]).sum())
    if duplicate_keys:
        issues.append(
            ValidationIssue(
                "unique_engine_cycle",
                duplicate_keys,
                "An engine can have only one snapshot per cycle.",
            )
        )

    bad_sequences = 0
    for _, engine_rows in frame.groupby("engine_id", sort=False):
        cycles = sorted(engine_rows["cycle"].dropna().astype(int).unique())
        if cycles and cycles != list(range(1, max(cycles) + 1)):
            bad_sequences += 1
    if bad_sequences:
        issues.append(
            ValidationIssue(
                "consecutive_cycles",
                bad_sequences,
                "Each engine must contain every cycle from 1 to its latest cycle.",
            )
        )

    return ValidationReport(
        dataset=dataset,
        rows=len(frame),
        engines=int(frame["engine_id"].nunique()),
        issues=tuple(issues),
    )


def validate_fd001_bundle(
    train: pd.DataFrame, test: pd.DataFrame, truth: pd.Series
) -> tuple[ValidationReport, ValidationReport, tuple[ValidationIssue, ...]]:
    """Validate train/test frames plus their cross-file truth relationship."""

    train_report = validate_sensor_frame(train, "train_FD001")
    test_report = validate_sensor_frame(test, "test_FD001")
    truth_issues: list[ValidationIssue] = []

    expected = int(test["engine_id"].nunique()) if "engine_id" in test else 0
    if len(truth) != expected:
        truth_issues.append(
            ValidationIssue(
                "truth_engine_count",
                abs(len(truth) - expected),
                f"Expected {expected} truth rows, found {len(truth)}.",
            )
        )
    negative = int((truth < 0).sum())
    if negative:
        truth_issues.append(
            ValidationIssue(
                "non_negative_truth", negative, "Remaining life cannot be negative."
            )
        )
    return train_report, test_report, tuple(truth_issues)


def raise_for_invalid(
    train_report: ValidationReport,
    test_report: ValidationReport,
    truth_issues: tuple[ValidationIssue, ...],
) -> None:
    problems = [*train_report.issues, *test_report.issues, *truth_issues]
    if problems:
        message = "; ".join(f"{item.rule}: {item.detail}" for item in problems)
        raise ValueError(f"FD001 validation failed: {message}")

