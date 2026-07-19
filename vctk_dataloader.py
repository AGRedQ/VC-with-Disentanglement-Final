from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset


def import_config():
	import yaml

	config_path = Path(__file__).parent / "config.yaml"
	with open(config_path, "r") as f:
		config = yaml.safe_load(f)
	return config


class VCTKDataset(Dataset):
	def __init__(
		self,
		root=None,
		sample_rate=None,
		mic=None,
	):
		config = import_config()
		self.root = Path(root or config.get("vctk_root", "datasets/vctk/wav48_silence_trimmed"))
		self.sample_rate = sample_rate or config.get("sample_rate", 16000)
		self.mic = mic or config.get("mic", "mic1")
		self.files = sorted(self.root.glob(f"p*/*_{self.mic}.flac"))

		if not self.files:
			raise FileNotFoundError(f"No VCTK files found under {self.root} for {self.mic}")

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		audio_path = self.files[index]
		stem_parts = audio_path.stem.split("_")
		waveform_array, original_sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
		waveform = torch.from_numpy(waveform_array).transpose(0, 1)

		if waveform.shape[0] > 1:
			waveform = waveform.mean(dim=0, keepdim=True)

		if original_sample_rate != self.sample_rate:
			waveform = torchaudio.functional.resample(
				waveform,
				orig_freq=original_sample_rate,
				new_freq=self.sample_rate,
			)

		waveform = waveform.squeeze(0).float()

		return {
			"waveform": waveform,
			"sample_rate": self.sample_rate,
			"speaker_id": audio_path.parent.name,
			"utterance_id": stem_parts[1],
			"mic": stem_parts[2],
			"path": str(audio_path),
		}


if __name__ == "__main__":
	dataset = VCTKDataset()
	sample = dataset[1]

	print("Dataset size:", len(dataset))
	print("Speaker ID:", sample["speaker_id"])
	print("Utterance ID:", sample["utterance_id"])
	print("Waveform shape:", sample["waveform"].shape)
	print("Sample rate:", sample["sample_rate"])
	print("Path:", sample["path"])
