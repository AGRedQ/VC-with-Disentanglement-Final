from pathlib import Path
import hashlib

import torch
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


CONFIG_PATH = Path(__file__).parent / "configs.yaml"


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


def speaker_id_from_sample_id(sample_id):
    return sample_id.split("_", 1)[0]


def sorted_split_keys(keys, seed):
    return sorted(
        keys,
        key=lambda key: hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest(),
    )


def split_sample_ids(sample_ids, split, config):
    if split is None or split == "all":
        return sample_ids

    split_config = get_nested(config, "dataset", "split", default={}) or {}
    train_ratio = float(split_config.get("train", 0.8))
    val_ratio = float(split_config.get("val", 0.1))
    test_ratio = float(split_config.get("test", 0.1))
    seed = int(split_config.get("seed", 1234))
    split_by = split_config.get("split_by", "speaker")

    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be one of train, val, test, all, or None; got {split!r}")
    if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("dataset split ratios must be non-negative")

    ratio_total = train_ratio + val_ratio + test_ratio
    if ratio_total <= 0:
        raise ValueError("at least one dataset split ratio must be greater than zero")

    train_ratio /= ratio_total
    val_ratio /= ratio_total

    groups = {}
    for sample_id in sample_ids:
        group_key = speaker_id_from_sample_id(sample_id) if split_by == "speaker" else sample_id
        groups.setdefault(group_key, []).append(sample_id)

    group_keys = sorted_split_keys(groups.keys(), seed)
    num_groups = len(group_keys)
    num_train = round(num_groups * train_ratio)
    num_val = round(num_groups * val_ratio)

    if num_groups >= 3:
        num_train = min(max(num_train, 1), num_groups - 2)
        num_val = min(max(num_val, 1), num_groups - num_train - 1)
    else:
        num_train = max(1, min(num_train, num_groups))
        num_val = max(0, min(num_val, num_groups - num_train))

    split_to_keys = {
        "train": group_keys[:num_train],
        "val": group_keys[num_train:num_train + num_val],
        "test": group_keys[num_train + num_val:],
    }

    selected_ids = []
    for group_key in split_to_keys[split]:
        selected_ids.extend(groups[group_key])
    return sorted(selected_ids)


def squeeze_content(content_latent):
    if content_latent.dim() == 3 and content_latent.size(0) == 1:
        content_latent = content_latent.squeeze(0)
    if content_latent.dim() != 2:
        raise ValueError(f"content_latent must resolve to [T, C], got {content_latent.shape}")
    return content_latent.float()


def squeeze_speaker(speaker_latent):
    speaker_latent = speaker_latent.squeeze()
    if speaker_latent.dim() != 1:
        raise ValueError(f"speaker_latent must resolve to [S], got {speaker_latent.shape}")
    return speaker_latent.float()


def squeeze_mel(mel):
    if mel.dim() == 3 and mel.size(0) == 1:
        mel = mel.squeeze(0)
    if mel.dim() != 2:
        raise ValueError(f"mel must resolve to [T, M], got {mel.shape}")
    return mel.float()


class VCDataset(Dataset):
    def __init__(
        self,
        content_dir=None,
        speaker_dir=None,
        mel_dir=None,
        config=None,
        limit=None,
        split="all",
    ):
        self.config = config or load_config()
        self.split = split
        self.content_dir = Path(content_dir or get_nested(
            self.config,
            "precomputed",
            "content_latent_dir",
            default="datasets/precomputed/contents",
        ))
        self.speaker_dir = Path(speaker_dir or get_nested(
            self.config,
            "precomputed",
            "speaker_latent_dir",
            default="datasets/precomputed/speakers",
        ))
        self.mel_dir = Path(mel_dir or get_nested(
            self.config,
            "precomputed",
            "mel_dir",
            default="datasets/precomputed/mels",
        ))

        content_paths = sorted(self.content_dir.glob("*.pt"))
        sample_ids = []
        for content_path in content_paths:
            sample_id = content_path.name
            if (self.speaker_dir / sample_id).exists() and (self.mel_dir / sample_id).exists():
                sample_ids.append(sample_id)

        sample_ids = split_sample_ids(sample_ids, split, self.config)

        if limit is not None:
            sample_ids = sample_ids[:limit]

        if not sample_ids:
            raise FileNotFoundError(
                "No paired content/speaker/mel samples found. "
                f"Checked {self.content_dir}, {self.speaker_dir}, and {self.mel_dir}."
            )

        self.sample_ids = sample_ids

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        content_payload = torch.load(self.content_dir / sample_id, map_location="cpu")
        speaker_payload = torch.load(self.speaker_dir / sample_id, map_location="cpu")
        mel_payload = torch.load(self.mel_dir / sample_id, map_location="cpu")

        content = squeeze_content(content_payload["content_latent"])
        speaker = squeeze_speaker(speaker_payload["speaker_latent"])
        mel = squeeze_mel(mel_payload["mel"])

        time_steps = min(content.size(0), mel.size(0))
        content = content[:time_steps]
        mel = mel[:time_steps]

        return {
            "sample_id": sample_id,
            "content": content,
            "speaker": speaker,
            "mel": mel,
            "length": time_steps,
        }


def vc_collate_fn(samples):
    contents = [sample["content"] for sample in samples]
    speakers = torch.stack([sample["speaker"] for sample in samples], dim=0)
    mels = [sample["mel"] for sample in samples]
    lengths = torch.tensor([sample["length"] for sample in samples], dtype=torch.long)

    content_batch = pad_sequence(contents, batch_first=True)
    mel_batch = pad_sequence(mels, batch_first=True)
    max_length = content_batch.size(1)
    mask = torch.arange(max_length).unsqueeze(0) < lengths.unsqueeze(1)

    return {
        "sample_ids": [sample["sample_id"] for sample in samples],
        "content": content_batch,
        "speaker": speakers,
        "mel": mel_batch,
        "lengths": lengths,
        "mask": mask,
    }


if __name__ == "__main__":
    for split_name in ["train", "val", "test"]:
        print(f"{split_name} dataset size:", len(VCDataset(split=split_name)))

    dataset = VCDataset(split="train", limit=4)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=vc_collate_fn)
    batch = next(iter(dataloader))

    print("Content batch:", batch["content"].shape)
    print("Speaker batch:", batch["speaker"].shape)
    print("Mel batch:", batch["mel"].shape)
    print("Mask:", batch["mask"].shape)
