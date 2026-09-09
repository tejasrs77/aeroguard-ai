"""Download and unpack the official NASA C-MAPSS archive reproducibly."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

from aeroguard.config import (
    ARCHIVE_NAME,
    FD001_FILES,
    NASA_ARCHIVE_URL,
    NASA_DATASET_PAGE,
    RAW_DIR,
    ensure_directories,
)


def sha256_file(path: Path) -> str:
    """Return a stable fingerprint for a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_to(url: str, destination: Path) -> None:
    with requests.get(
        url,
        headers={"User-Agent": "AeroGuard-AI/0.1 educational-project"},
        timeout=(15, 120),
        stream=True,
    ) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    output.write(block)


def extract_fd001(archive_path: Path, raw_dir: Path = RAW_DIR) -> list[Path]:
    """Extract only the three files required for the FD001 experiment."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    with zipfile.ZipFile(archive_path) as archive:
        members_by_name = {Path(member.filename).name: member for member in archive.infolist()}
        missing = sorted(set(FD001_FILES) - set(members_by_name))
        if missing:
            raise ValueError(f"NASA archive is missing required files: {missing}")

        for filename in FD001_FILES:
            destination = raw_dir / filename
            with archive.open(members_by_name[filename]) as source:
                with destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
            extracted.append(destination)

    return extracted


def download_dataset(
    raw_dir: Path = RAW_DIR,
    *,
    force: bool = False,
    url: str = NASA_ARCHIVE_URL,
) -> dict[str, object]:
    """Download, fingerprint, extract, and describe the C-MAPSS archive.

    An existing archive is reused unless ``force`` is true. Downloading to a
    temporary file first prevents an interrupted request from leaving a
    corrupted file with the final archive name.
    """

    ensure_directories()
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / ARCHIVE_NAME

    if force or not archive_path.exists():
        with tempfile.NamedTemporaryFile(
            prefix="cmapss-", suffix=".zip", dir=raw_dir, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            _download_to(url, temporary_path)
            if not zipfile.is_zipfile(temporary_path):
                raise ValueError("Downloaded file is not a valid ZIP archive")
            temporary_path.replace(archive_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    if not zipfile.is_zipfile(archive_path):
        raise ValueError(f"Existing archive is not a valid ZIP file: {archive_path}")

    extracted = extract_fd001(archive_path, raw_dir)
    manifest: dict[str, object] = {
        "dataset": "NASA C-MAPSS Jet Engine Simulated Data",
        "subset": "FD001",
        "dataset_page": NASA_DATASET_PAGE,
        "download_url": url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": ARCHIVE_NAME,
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "extracted_files": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in extracted
        ],
    }
    manifest_path = raw_dir / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
