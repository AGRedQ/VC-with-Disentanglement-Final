import argparse
import json
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from contentvec import encode_content, load_content_encoder
from mel_spectrogram import load_waveform
from vc_dataset import load_config


CONFIG_PATH = Path(__file__).parent / "configs.yaml"


def get_nested(config, *keys, default=None):
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def iter_vctk_files(root, mic):
    root = Path(root)
    return sorted(root.glob(f"p*/*_{mic}.flac"))


def sample_id_from_audio_path(audio_path):
    stem_parts = audio_path.stem.split("_")
    speaker_id = audio_path.parent.name
    utterance_id = stem_parts[1]
    mic = stem_parts[2]
    return speaker_id, utterance_id, mic, f"{speaker_id}_{utterance_id}_{mic}"


def save_progress(progress_path, progress, output_dir):
    progress["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    progress["saved_files_total"] = len(list(output_dir.glob("*.pt")))
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def precompute_content_latents(
    audio_paths,
    output_dir,
    config=None,
    progress_path=None,
    progress_every=25,
    limit=None,
    device=None,
):
    config = config or load_config(CONFIG_PATH)
    sample_rate = get_nested(config, "audio", "sampling_rate", default=16000)
    model_id = get_nested(config, "contentvec", "model_id", default="lengyue233/content-vec-best")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(progress_path or output_dir / "content_progress.json")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    content_encoder = load_content_encoder(device=device)
    total = len(audio_paths) if limit is None else min(len(audio_paths), limit)
    progress = {
        "total": total,
        "dataset_size": len(audio_paths),
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

    content_paths = []

    try:
        for index, audio_path in enumerate(tqdm(audio_paths, desc="Computing ContentVec latents", total=total)):
            if limit is not None and index >= limit:
                break

            speaker_id, utterance_id, mic, sample_id = sample_id_from_audio_path(audio_path)
            content_path = output_dir / f"{sample_id}.pt"

            progress["last_index"] = index
            progress["last_file"] = content_path.name

            if content_path.exists():
                content_paths.append(content_path)
                progress["skipped_existing_this_run"] += 1
                if (index + 1) % progress_every == 0:
                    save_progress(progress_path, progress, output_dir)
                continue

            waveform = load_waveform(audio_path, sample_rate=sample_rate)
            content = encode_content(content_encoder, waveform, device=device).squeeze(0).cpu()

            torch.save(
                {
                    "content": content,
                    "speaker_id": speaker_id,
                    "utterance_id": utterance_id,
                    "mic": mic,
                    "sample_rate": sample_rate,
                    "model_id": model_id,
                    "path": str(audio_path),
                },
                content_path,
            )

            content_paths.append(content_path)
            progress["processed_this_run"] += 1

            if (index + 1) % progress_every == 0:
                save_progress(progress_path, progress, output_dir)

    except KeyboardInterrupt:
        progress["interrupted"] = True
        save_progress(progress_path, progress, output_dir)
        raise

    progress["completed"] = True
    save_progress(progress_path, progress, output_dir)
    return content_paths


def print_progress(output_dir):
    output_dir = Path(output_dir)
    progress_path = output_dir / "content_progress.json"
    saved_count = len(list(output_dir.glob("*.pt"))) if output_dir.exists() else 0

    print("Saved content latent files:", saved_count)
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
        print("No content progress file yet.")


def parse_args():
    config = load_config(CONFIG_PATH)
    parser = argparse.ArgumentParser(description="Precompute ContentVec latents for VCTK.")
    parser.add_argument("--root", default=get_nested(config, "dataset", "vctk_root", default="datasets/vctk/wav48_silence_trimmed"))
    parser.add_argument("--mic", default=get_nested(config, "dataset", "mic", default="mic1"))
    parser.add_argument("--output-dir", default=get_nested(config, "precomputed", "content_latent_dir", default="datasets/precomputed/contents"))
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda, cpu, or omitted for auto.")
    parser.add_argument("--status", action="store_true", help="Print progress and exit without computing latents.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.status:
        print_progress(args.output_dir)
        return

    config = load_config(CONFIG_PATH)
    audio_paths = iter_vctk_files(args.root, args.mic)
    if not audio_paths:
        raise FileNotFoundError(f"No VCTK files found under {args.root} for {args.mic}")

    print("Audio files:", len(audio_paths))
    print("Content latent output dir:", args.output_dir)
    print("Using ContentVec device:", args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    content_paths = precompute_content_latents(
        audio_paths=audio_paths,
        output_dir=args.output_dir,
        config=config,
        progress_every=args.progress_every,
        limit=args.limit,
        device=args.device,
    )
    print("Computed/skipped content latent files:", len(content_paths))


if __name__ == "__main__":
    main()