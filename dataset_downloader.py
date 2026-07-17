"""Download datasets used by this project."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


VCTK_DATASET = "kynthesis/vctk-corpus"
VCTK_URL = "https://datashare.ed.ac.uk/bitstreams/535f4286-e54c-4038-838c-a02285e32cb2/download"
VCTK_ZIP_NAME = "VCTK-Corpus-0.92.zip"
VCTK_DESTINATION = Path("datasets") / "vctk"


def download_vctk_official(destination: Path = VCTK_DESTINATION, force: bool = False) -> None:
	destination.mkdir(parents=True, exist_ok=True)
	archive_path = destination / VCTK_ZIP_NAME

	command = [
		"curl.exe",
		"-L",
		"--fail",
		"--continue-at",
		"-",
		"--output",
		str(archive_path),
		VCTK_URL,
	]

	if force and archive_path.exists():
		archive_path.unlink()

	subprocess.run(command, check=True)


def download_vctk_kaggle(destination: Path = VCTK_DESTINATION, force: bool = False) -> None:
	destination.mkdir(parents=True, exist_ok=True)

	if any(destination.iterdir()) and not force:
		print(f"{destination} is not empty. Use --force to download again.")
		return

	command = [
		sys.executable,
		"-m",
		"kaggle",
		"datasets",
		"download",
		"-d",
		VCTK_DATASET,
		"-p",
		str(destination),
		
	]

	if force:
		command.append("--force")

	try:
		subprocess.run(command, check=True)
	except ModuleNotFoundError as error:
		raise SystemExit("Install the Kaggle package first: pip install kaggle") from error
	except subprocess.CalledProcessError as error:
		raise SystemExit(
			"Kaggle download failed. Make sure your Kaggle API token is configured at "
			"%USERPROFILE%\\.kaggle\\kaggle.json or set KAGGLE_USERNAME and KAGGLE_KEY."
		) from error


def download_vctk(destination: Path = VCTK_DESTINATION, force: bool = False, source: str = "official") -> None:
	if source == "official":
		download_vctk_official(destination=destination, force=force)
	elif source == "kaggle":
		download_vctk_kaggle(destination=destination, force=force)
	else:
		raise ValueError(f"Unsupported VCTK source: {source}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Download the VCTK Corpus 0.92.")
	parser.add_argument("--force", action="store_true", help="Restart the download from the beginning.")
	parser.add_argument(
		"--source",
		choices=("official", "kaggle"),
		default="official",
		help="Download from the resumable official URL or from Kaggle.",
	)
	parser.add_argument(
		"--destination",
		type=Path,
		default=VCTK_DESTINATION,
		help="Directory to place the VCTK zip file.",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	download_vctk(destination=args.destination, force=args.force, source=args.source)
