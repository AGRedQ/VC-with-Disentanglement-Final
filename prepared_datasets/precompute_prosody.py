import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from tqdm.auto import tqdm

from mel_spectrogram import get_nested, iter_vctk_files, load_config, load_waveform


CONFIG_PATH = Path(__file__).parent.parent / "configs.yaml"


def resize_1d(values, target_frames):
    values = values.float().view(1, 1, -1)
    values = F.interpolate(values, size=target_frames, mode="linear", align_corners=False)
    return values.view(target_frames)


def compute_energy(waveform, n_fft, hop_length, target_frames):
    padding = n_fft // 2
    padded = F.pad(waveform.view(1, 1, -1), (padding, padding), mode="reflect").view(-1)
    frames = padded.unfold(0, n_fft, hop_length)
    energy = torch.sqrt(torch.clamp(frames.pow(2).mean(dim=-1), min=1.0e-12))
    if energy.numel() != target_frames:
        energy = resize_1d(energy, target_frames)
    log_energy = torch.log(torch.clamp(energy, min=1.0e-5))
    return (log_energy - log_energy.mean()) / log_energy.std().clamp_min(1.0e-5)


def compute_f0_vuv(waveform, sample_rate, hop_length, target_frames, freq_low=50, freq_high=600):
    frame_time = hop_length / sample_rate
    f0 = torchaudio.functional.detect_pitch_frequency(
        waveform.unsqueeze(0),
        sample_rate=sample_rate,
        frame_time=frame_time,
        freq_low=freq_low,
        freq_high=freq_high,
    ).squeeze(0)

    if f0.numel() != target_frames:
        f0 = resize_1d(f0, target_frames)

    vuv = (f0 > 0).float()
    voiced = vuv > 0.5
    log_f0_norm = torch.zeros_like(f0)
    if voiced.any():
        log_f0 = torch.log(torch.clamp(f0[voiced], min=1.0))
        log_f0_norm[voiced] = (log_f0 - log_f0.mean()) / log_f0.std().clamp_min(1.0e-5)

    return log_f0_norm, vuv, f0


def sample_id_from_audio_path(audio_path):
    stem_parts = audio_path.stem.split("_")
    speaker_id = audio_path.parent.name
    utterance_id = stem_parts[1]
    mic = stem_parts[2]
    return f"{speaker_id}_{utterance_id}_{mic}.pt", speaker_id, utterance_id, mic


def save_progress(progress_path, progress, output_dir):
    progress["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    progress["saved_files_total"] = len(list(output_dir.glob("*.pt")))
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def compute_prosody_features(
    audio_paths,
    output_dir,
    config=None,
    progress_path=None,
    progress_every=25,
    limit=None,
):
    config = config or load_config(CONFIG_PATH)
    sample_rate = get_nested(config, "audio", "sampling_rate", default=16000)
    hop_length = get_nested(config, "mel", "hop_length", default=320)
    n_fft = get_nested(config, "mel", "n_fft", default=1024)
    mel_dir = Path(get_nested(config, "precomputed", "mel_dir", default="datasets/precomputed/mels"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(progress_path or output_dir / "prosody_progress.json")

    total = len(audio_paths) if limit is None else min(len(audio_paths), limit)
    progress = {
        "total": total,
        "dataset_size": len(audio_paths),
        "existing_before_start": len(list(output_dir.glob("*.pt"))),
        "processed_this_run": 0,
        "skipped_existing_this_run": 0,
        "skipped_missing_mel_this_run": 0,
        "last_index": None,
        "last_file": None,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": None,
        "completed": False,
    }
    save_progress(progress_path, progress, output_dir)

    prosody_paths = []

    try:
        for index, audio_path in enumerate(tqdm(audio_paths, desc="Computing prosody", total=total)):
            if limit is not None and index >= limit:
                break

            sample_id, speaker_id, utterance_id, mic = sample_id_from_audio_path(audio_path)
            prosody_path = output_dir / sample_id
            mel_path = mel_dir / sample_id
            progress["last_index"] = index
            progress["last_file"] = prosody_path.name

            if prosody_path.exists():
                prosody_paths.append(prosody_path)
                progress["skipped_existing_this_run"] += 1
                if (index + 1) % progress_every == 0:
                    save_progress(progress_path, progress, output_dir)
                continue

            if not mel_path.exists():
                progress["skipped_missing_mel_this_run"] += 1
                if (index + 1) % progress_every == 0:
                    save_progress(progress_path, progress, output_dir)
                continue

            mel_payload = torch.load(mel_path, map_location="cpu")
            target_frames = int(mel_payload["mel"].shape[0])
            waveform = load_waveform(audio_path, sample_rate=sample_rate)
            log_f0_norm, vuv, f0_hz = compute_f0_vuv(
                waveform,
                sample_rate=sample_rate,
                hop_length=hop_length,
                target_frames=target_frames,
            )
            energy_norm = compute_energy(
                waveform,
                n_fft=n_fft,
                hop_length=hop_length,
                target_frames=target_frames,
            )
            prosody = torch.stack([log_f0_norm, vuv, energy_norm], dim=-1)

            torch.save(
                {
                    "prosody": prosody.cpu(),
                    "log_f0_norm": log_f0_norm.cpu(),
                    "vuv": vuv.cpu(),
                    "energy_norm": energy_norm.cpu(),
                    "f0_hz": f0_hz.cpu(),
                    "feature_names": ["log_f0_norm", "vuv", "energy_norm"],
                    "speaker_id": speaker_id,
                    "utterance_id": utterance_id,
                    "mic": mic,
                    "sample_rate": sample_rate,
                    "hop_length": hop_length,
                    "path": str(audio_path),
                },
                prosody_path,
            )

            prosody_paths.append(prosody_path)
            progress["processed_this_run"] += 1

            if (index + 1) % progress_every == 0:
                save_progress(progress_path, progress, output_dir)

    except KeyboardInterrupt:
        progress["interrupted"] = True
        save_progress(progress_path, progress, output_dir)
        raise

    progress["completed"] = True
    save_progress(progress_path, progress, output_dir)
    return prosody_paths


def print_progress(output_dir):
    output_dir = Path(output_dir)
    progress_path = output_dir / "prosody_progress.json"
    saved_count = len(list(output_dir.glob("*.pt"))) if output_dir.exists() else 0

    print("Saved prosody files:", saved_count)
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        print("Progress file:", progress_path)
        print("Total target:", progress.get("total"))
        print("Dataset size:", progress.get("dataset_size"))
        print("Processed this run:", progress.get("processed_this_run"))
        print("Skipped existing this run:", progress.get("skipped_existing_this_run"))
        print("Skipped missing mel this run:", progress.get("skipped_missing_mel_this_run"))
        print("Last index:", progress.get("last_index"))
        print("Last file:", progress.get("last_file"))
        print("Completed:", progress.get("completed"))
    else:
        print("No prosody progress file yet.")


def parse_args():
    config = load_config(CONFIG_PATH)
    parser = argparse.ArgumentParser(description="Precompute F0/VUV/energy prosody features for VCTK.")
    parser.add_argument("--root", default=get_nested(config, "dataset", "vctk_root", default="datasets/vctk/wav48_silence_trimmed"))
    parser.add_argument("--mic", default=get_nested(config, "dataset", "mic", default="mic1"))
    parser.add_argument("--output-dir", default=get_nested(config, "precomputed", "prosody_dir", default="datasets/precomputed/prosody"))
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--status", action="store_true", help="Print progress and exit without computing features.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.status:
        print_progress(args.output_dir)
        return

    config = load_config(CONFIG_PATH)
    audio_paths = iter_vctk_files(args.root, args.mic)
    print(f"Found {len(audio_paths)} audio files under {args.root} for {args.mic}.")
    paths = compute_prosody_features(
        audio_paths,
        output_dir=args.output_dir,
        config=config,
        progress_every=args.progress_every,
        limit=args.limit,
    )
    print(f"Prosody files available in this run: {len(paths)}")
    print_progress(args.output_dir)


if __name__ == "__main__":
    main()
