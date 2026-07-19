
"""
AutoVC-style decoder for reconstructing mel spectrograms from content and speaker latents.
"""

import torch
import torch.nn as nn


class ConvNorm(nn.Module):
	def __init__(self, input_dim, output_dim, kernel_size=5, padding=None):
		super().__init__()
		if padding is None:
			padding = (kernel_size - 1) // 2

		self.conv = nn.Conv1d(
			input_dim,
			output_dim,
			kernel_size=kernel_size,
			stride=1,
			padding=padding,
			dilation=1,
		)

	def forward(self, inputs):
		return self.conv(inputs)


class AutoVCDecoder(nn.Module):
	def __init__(
		self,
		content_dim=768, # [B, T, 768] for content latent
		speaker_dim=192, # [B, 1, 192] for speaker latent
		mel_dim=80, 
		prenet_dim=512,
		lstm_dim=1024,
		num_conv_layers=3,
		conv_kernel_size=5,
		dropout=0.5,
	):
		super().__init__()
		input_dim = content_dim + speaker_dim

		self.lstm1 = nn.LSTM(
			input_size=input_dim,
			hidden_size=prenet_dim,
			num_layers=1,
			batch_first=True,
			bidirectional=False,
		)

		conv_layers = []
		for _ in range(num_conv_layers):
			conv_layers.append(
				nn.Sequential(
					ConvNorm(prenet_dim, prenet_dim, kernel_size=conv_kernel_size),
					nn.BatchNorm1d(prenet_dim),
					nn.ReLU(),
					nn.Dropout(dropout),
				)
			)
		self.convolutions = nn.ModuleList(conv_layers)

		self.lstm2 = nn.LSTM(
			input_size=prenet_dim,
			hidden_size=lstm_dim,
			num_layers=2,
			batch_first=True,
			bidirectional=False,
			dropout=dropout,
		)
		self.linear_projection = nn.Linear(lstm_dim, mel_dim)

	def forward(self, content_latent, speaker_latent):
		# Checking input dim
		if content_latent.dim() == 4 and content_latent.size(1) == 1: 
			content_latent = content_latent.squeeze(1)
		if content_latent.dim() != 3:
			raise ValueError(f"content_latent must have shape [B, T, C], got {content_latent.shape}")

		speaker_latent = speaker_latent.squeeze()
		if speaker_latent.dim() == 1:
			speaker_latent = speaker_latent.unsqueeze(0)
		if speaker_latent.dim() != 2:
			raise ValueError(f"speaker_latent must have shape [B, S], got {speaker_latent.shape}")

		batch_size, time_steps, _ = content_latent.shape # [B, T, C]
		if speaker_latent.size(0) != batch_size:
			raise ValueError(
				"content_latent and speaker_latent batch sizes must match: "
				f"{batch_size} != {speaker_latent.size(0)}"
			)

		speaker_latent = speaker_latent.unsqueeze(1).expand(-1, time_steps, -1)
		decoder_input = torch.cat([content_latent, speaker_latent], dim=-1)

		output, _ = self.lstm1(decoder_input)
		output = output.transpose(1, 2)
		for convolution in self.convolutions:
			output = convolution(output)
		output = output.transpose(1, 2)
		output, _ = self.lstm2(output)
		mel = self.linear_projection(output)
		return mel


if __name__ == "__main__":
	decoder = AutoVCDecoder()
	content = torch.randn(2, 100, 768)
	speaker = torch.randn(2, 1, 192)
	mel = decoder(content, speaker)
	print("Output mel shape:", mel.shape)