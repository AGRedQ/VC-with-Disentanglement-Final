from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset


class VCTKDataset(Dataset):
	def __init__(
		self,
		root="datasets/vctk/wav48_silence_trimmed",
		sample_rate=16000,
		mic="mic1",
	):
		self.root = Path(root)
		self.sample_rate = sample_rate
		self.mic = mic
		self.files = sorted(self.root.glob(f"p*/*_{mic}.flac"))

		if not self.files:
			raise FileNotFoundError(f"No VCTK files found under {self.root} for {mic}")

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
