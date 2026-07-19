import argparse
import json
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from prepared_datasets.ecapa import encode_speaker, get_default_device, load_speaker_encoder
from prepared_datasets.vctk_dataloader import VCTKDataset


def save_progress(progress_path, progress, output_dir):
    progress["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    progress["saved_files_total"] = len(list(output_dir.glob("*.pt")))
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def compute_speaker_latents(
    dataset,
    speaker_encoder,
    output_dir="datasets/precomputed/speakers",
    progress_path=None,
    device=None,
    progress_every=25,
    limit=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(progress_path or output_dir / "speaker_latent_progress.json")
    device = torch.device(device or get_default_device())

    total = len(dataset) if limit is None else min(len(dataset), limit)
    progress = {
        "total": total,
        "dataset_size": len(dataset),
        "existing_before_start": len(list(output_dir.glob("*.pt"))),
        "processed_this_run": 0,
        "skipped_existing_this_run": 0,
        "last_index": None,
        "last_file": None,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": None,
        "completed": False,
    }
    save_progress(progress_path, progress, output_dir)

    latent_paths = []

    try:
        for index, sample in enumerate(tqdm(dataset, desc="Computing speaker latents", total=total)):
            if limit is not None and index >= limit:
                break

            speaker_id = str(sample["speaker_id"])
            utterance_id = str(sample["utterance_id"])
            mic = str(sample["mic"])
            latent_path = output_dir / f"{speaker_id}_{utterance_id}_{mic}.pt"

            progress["last_index"] = index
            progress["last_file"] = latent_path.name

            if latent_path.exists():
                latent_paths.append(latent_path)
                progress["skipped_existing_this_run"] += 1
                if (index + 1) % progress_every == 0:
                    save_progress(progress_path, progress, output_dir)
                continue

            with torch.inference_mode():
                latent = encode_speaker(speaker_encoder, sample["waveform"], device=device)

            torch.save(
                {
                    "speaker_latent": latent.cpu() if hasattr(latent, "cpu") else latent,
                    "speaker_id": sample["speaker_id"],
                    "utterance_id": sample["utterance_id"],
                    "mic": sample["mic"],
                    "sample_rate": sample["sample_rate"],
                    "path": sample["path"],
                },
                latent_path,
            )

            latent_paths.append(latent_path)
            progress["processed_this_run"] += 1

            if (index + 1) % progress_every == 0:
                save_progress(progress_path, progress, output_dir)

    except KeyboardInterrupt:
        progress["interrupted"] = True
        save_progress(progress_path, progress, output_dir)
        raise

    progress["completed"] = True
    save_progress(progress_path, progress, output_dir)
    return latent_paths


def print_progress(output_dir):
    output_dir = Path(output_dir)
    progress_path = output_dir / "speaker_latent_progress.json"
    saved_count = len(list(output_dir.glob("*.pt"))) if output_dir.exists() else 0

    print("Saved speaker latent files:", saved_count)
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        print("Progress file:", progress_path)
        print("Total target:", progress.get("total"))
        print("Dataset size:", progress.get("dataset_size"))
        print("Processed this run:", progress.get("processed_this_run"))
        print("Skipped existing this run:", progress.get("skipped_existing_this_run"))
        print("Last index:", progress.get("last_index"))
        print("Last file:", progress.get("last_file"))
        print("Completed:", progress.get("completed"))
    else:
        print("No speaker progress file yet.")


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute ECAPA speaker latents for VCTK.")
    parser.add_argument("--root", default="datasets/vctk/wav48_silence_trimmed")
    parser.add_argument("--mic", default="mic1")
    parser.add_argument("--output-dir", default="datasets/precomputed/speakers")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--status", action="store_true", help="Print progress and exit without loading ECAPA.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.status:
        print_progress(args.output_dir)
        return

    device = get_default_device()
    print("Torch CUDA available:", torch.cuda.is_available())
    print("Using speaker device:", device)

    dataset = VCTKDataset(root=args.root, mic=args.mic)
    print("Dataset size:", len(dataset))

    speaker_encoder = load_speaker_encoder(device=device)
    print("Speaker Encoder Model:", type(speaker_encoder).__name__)

    latent_paths = compute_speaker_latents(
        dataset=dataset,
        speaker_encoder=speaker_encoder,
        output_dir=args.output_dir,
        device=device,
        progress_every=args.progress_every,
        limit=args.limit,
    )
    print("Computed/skipped speaker latents:", len(latent_paths))


if __name__ == "__main__":
    main()