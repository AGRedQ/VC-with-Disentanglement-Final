"""
AutoVC-style decoder and training utilities for mel reconstruction.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from vc_dataset import VCDataset, load_config, vc_collate_fn

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def masked_l1_loss(pred_mel, target_mel, mask):
    mask = mask.to(pred_mel.device).unsqueeze(-1)
    target_mel = target_mel.to(pred_mel.device)
    loss = (pred_mel - target_mel).abs() * mask
    denominator = (mask.sum() * pred_mel.size(-1)).clamp_min(1.0)
    return loss.sum() / denominator


def move_batch_to_device(batch, device):
    return {
        "sample_ids": batch["sample_ids"],
        "content": batch["content"].to(device),
        "speaker": batch["speaker"].to(device),
        "mel": batch["mel"].to(device),
        "lengths": batch["lengths"].to(device),
        "mask": batch["mask"].to(device),
    }


def train_one_epoch(model, dataloader, optimizer, device, description="Training"):
    model.train()
    total_loss = 0.0
    total_batches = 0

    progress = tqdm(dataloader, desc=description, leave=False)
    for batch in progress:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        pred_mel = model(batch["content"], batch["speaker"])
        loss = masked_l1_loss(pred_mel, batch["mel"], batch["mask"])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_batches, 1)


def save_checkpoint(model, optimizer, epoch, loss, config, checkpoint_dir="models/checkpoints"):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"decoder_epoch_{epoch:04d}.pt"

    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        checkpoint_path,
    )
    return checkpoint_path


def build_dataloader(dataset, training_config, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=shuffle,
        collate_fn=vc_collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def build_training_objects(config=None, dataset=None, model=None, device=None):
    config = config or load_config()
    training_config = config["training"]
    device = torch.device(device or get_device())
    dataset = dataset or VCDataset(config=config, split="train")
    model = (model or VCDecoder()).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    dataloader = build_dataloader(dataset, training_config, shuffle=True)

    return model, optimizer, dataloader, device


def run_training(config=None, dataset=None, model=None, device=None):
    config = config or load_config()
    training_config = config["training"]

    model, optimizer, train_loader, device = build_training_objects(
        config=config,
        dataset=dataset,
        model=model,
        device=device,
    )

    if training_config.get("smoke_test_training", False):
        smoke_dataset = VCDataset(config=config, split="train", limit=training_config["batch_size"])
        smoke_loader = build_dataloader(smoke_dataset, training_config, shuffle=False)
        smoke_loss = train_one_epoch(
            model,
            smoke_loader,
            optimizer,
            device,
            description="Smoke training",
        )
        print(f"Smoke train step complete on {device}. Loss: {smoke_loss:.4f}")
        return model, optimizer, {"smoke_loss": smoke_loss}

    history = []
    for epoch in range(1, training_config["num_epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        history.append({"epoch": epoch, "loss": train_loss})
        print(f"Epoch {epoch}/{training_config['num_epochs']} - loss: {train_loss:.4f}")

        if epoch % training_config["save_interval"] == 0:
            checkpoint_path = save_checkpoint(model, optimizer, epoch, train_loss, config)
            print(f"Saved checkpoint: {checkpoint_path}")

    return model, optimizer, {"history": history}


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


class VCDecoder(nn.Module):
    def __init__(
        self,
        content_dim=768, # [B, T, C] where C is the content latent dimension
        speaker_dim=192, # [B, S] where S is the speaker latent dimension
        mel_dim=80,      # [B, T, M] where M is the mel spectrogram dimension
        prenet_dim=512,
        lstm_dim=1024,
        num_conv_layers=3,
        conv_kernel_size=5,
        dropout=0.5,
    ):
        super().__init__()
        input_dim = content_dim + speaker_dim
        lstm1_output_dim = prenet_dim * 2

        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=prenet_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        conv_layers = []
        for layer_index in range(num_conv_layers):
            conv_input_dim = lstm1_output_dim if layer_index == 0 else prenet_dim
            conv_layers.append(
                nn.Sequential(
                    ConvNorm(conv_input_dim, prenet_dim, kernel_size=conv_kernel_size),
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
            bidirectional=True,
            dropout=dropout,
        )
        self.linear_projection = nn.Linear(2 * lstm_dim, mel_dim)

    def forward(self, content_latent, speaker_latent):
        if content_latent.dim() == 4 and content_latent.size(1) == 1:
            content_latent = content_latent.squeeze(1)
        if content_latent.dim() != 3:
            raise ValueError(f"content_latent must have shape [B, T, C], got {content_latent.shape}")

        speaker_latent = speaker_latent.squeeze()
        if speaker_latent.dim() == 1:
            speaker_latent = speaker_latent.unsqueeze(0)
        if speaker_latent.dim() != 2:
            raise ValueError(f"speaker_latent must have shape [B, S], got {speaker_latent.shape}")

        batch_size, time_steps, _ = content_latent.shape
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
    decoder = VCDecoder()
    content = torch.randn(2, 100, 768)
    speaker = torch.randn(2, 1, 192)
    mel = decoder(content, speaker)
    print("Output mel shape:", mel.shape)