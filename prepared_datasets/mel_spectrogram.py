import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
import yaml
from tqdm.auto import tqdm


CONFIG_PATH = Path(__file__).parent.parent / "configs.yaml"


def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def get_nested(config, *keys, default=None):
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def build_mel_transform(config=None, device=None):
    config = config or load_config()
    sample_rate = get_nested(config, "audio", "sampling_rate", default=16000)
    mel_config = config.get("mel", {})

    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=mel_config.get("n_fft", 1024),
        win_length=mel_config.get("win_length", 1024),
        hop_length=mel_config.get("hop_length", 320),
        n_mels=mel_config.get("n_mels", 80),
        f_min=mel_config.get("f_min", 0),
        f_max=mel_config.get("f_max", sample_rate // 2),
        power=mel_config.get("power", 1.0),
    )

    if device is not None:
        transform = transform.to(device)

    return transform


def load_waveform(audio_path, sample_rate):
    waveform_array, original_sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(waveform_array).transpose(0, 1)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if original_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=original_sample_rate,
            new_freq=sample_rate,
        )

    return waveform.squeeze(0).float()


def waveform_to_log_mel(waveform, mel_transform, log_min=1.0e-5, device=None):
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    if device is not None:
        waveform = waveform.to(device)

    mel = mel_transform(waveform)
    mel = torch.log(torch.clamp(mel, min=log_min))
    return mel.squeeze(0).transpose(0, 1)


def iter_vctk_files(root, mic):
    root = Path(root)
    return sorted(root.glob(f"p*/*_{mic}.flac"))


def save_progress(progress_path, progress, output_dir):
    progress["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    progress["saved_files_total"] = len(list(output_dir.glob("*.pt")))
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def compute_mel_spectrograms(
    audio_paths,
    output_dir,
    config=None,
    progress_path=None,
    progress_every=25,
    limit=None,
    device=None,
):
    config = config or load_config()
    sample_rate = get_nested(config, "audio", "sampling_rate", default=16000)
    log_min = get_nested(config, "mel", "log_min", default=1.0e-5)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(progress_path or output_dir / "mel_progress.json")
    device = torch.device(device or "cpu")
    mel_transform = build_mel_transform(config=config, device=device)

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

    mel_paths = []

    try:
        for index, audio_path in enumerate(tqdm(audio_paths, desc="Computing mel spectrograms", total=total)):
            if limit is not None and index >= limit:
                break

            stem_parts = audio_path.stem.split("_")
            speaker_id = audio_path.parent.name
            utterance_id = stem_parts[1]
            mic = stem_parts[2]
            mel_path = output_dir / f"{speaker_id}_{utterance_id}_{mic}.pt"

            progress["last_index"] = index
            progress["last_file"] = mel_path.name

            if mel_path.exists():
                mel_paths.append(mel_path)
                progress["skipped_existing_this_run"] += 1
                if (index + 1) % progress_every == 0:
                    save_progress(progress_path, progress, output_dir)
                continue

            waveform = load_waveform(audio_path, sample_rate=sample_rate)
            mel = waveform_to_log_mel(
                waveform,
                mel_transform=mel_transform,
                log_min=log_min,
                device=device,
            )

            torch.save(
                {
                    "mel": mel.cpu(),
                    "speaker_id": speaker_id,
                    "utterance_id": utterance_id,
                    "mic": mic,
                    "sample_rate": sample_rate,
                    "path": str(audio_path),
                },
                mel_path,
            )

            mel_paths.append(mel_path)
            progress["processed_this_run"] += 1

            if (index + 1) % progress_every == 0:
                save_progress(progress_path, progress, output_dir)

    except KeyboardInterrupt:
        progress["interrupted"] = True
        save_progress(progress_path, progress, output_dir)
        raise

    progress["completed"] = True
    save_progress(progress_path, progress, output_dir)
    return mel_paths


def print_progress(output_dir):
    output_dir = Path(output_dir)
    progress_path = output_dir / "mel_progress.json"
    saved_count = len(list(output_dir.glob("*.pt"))) if output_dir.exists() else 0

    print("Saved mel files:", saved_count)
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
        print("No mel progress file yet.")


def parse_args():
    config = load_config()
    parser = argparse.ArgumentParser(description="Precompute log mel spectrograms for VCTK.")
    parser.add_argument("--root", default=get_nested(config, "dataset", "vctk_root", default="datasets/vctk/wav48_silence_trimmed"))
    parser.add_argument("--mic", default=get_nested(config, "dataset", "mic", default="mic1"))
    parser.add_argument("--output-dir", default=get_nested(config, "precomputed", "mel_dir", default="datasets/precomputed/mels"))
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--status", action="store_true", help="Print progress and exit without computing mels.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.status:
        print_progress(args.output_dir)
        return

    config = load_config()
    audio_paths = iter_vctk_files(args.root, args.mic)
    if not audio_paths:
        raise FileNotFoundError(f"No VCTK files found under {args.root} for {args.mic}")

    print("Audio files:", len(audio_paths))
    print("Mel output dir:", args.output_dir)
    print("Using mel device:", args.device)

    mel_paths = compute_mel_spectrograms(
        audio_paths=audio_paths,
        output_dir=args.output_dir,
        config=config,
        progress_every=args.progress_every,
        limit=args.limit,
        device=args.device,
    )
    print("Computed/skipped mel files:", len(mel_paths))


if __name__ == "__main__":
    main()
"""
Used to create spectrograms from dataset. Will be used as X_pred
"""

def mel_spectrogram(waveform, sample_rate, n_fft=1024, hop_length=256, n_mels=80):
    import torchaudio.transforms as T

    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    mel_spec = mel_transform(waveform)
    return mel_spec