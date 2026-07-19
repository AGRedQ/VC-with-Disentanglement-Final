from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model_utils import get_device, masked_l1_loss, move_batch_to_device
from prepared_datasets.vc_dataset import VCDataset, load_config, vc_collate_fn

def train_one_epoch(model, dataloader, optimizer, device, description="Training"):
    model.train()
    total_loss = 0.0
    total_batches = 0

    progress = tqdm(dataloader, desc=description, leave=False)
    for batch in progress:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        coarse_mel, pred_mel = model(batch["content"], batch["speaker"], return_coarse=True)
        coarse_loss = masked_l1_loss(coarse_mel, batch["mel"], batch["mask"])
        refined_loss = masked_l1_loss(pred_mel, batch["mel"], batch["mask"])
        loss = coarse_loss + refined_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1
        progress.set_postfix(loss=f"{loss.item():.4f}", refined=f"{refined_loss.item():.4f}")

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate_loss(model, dataloader, device, description="Validation"):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    progress = tqdm(dataloader, desc=description, leave=False)
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        pred_mel = model(batch["content"], batch["speaker"])
        loss = masked_l1_loss(pred_mel, batch["mel"], batch["mask"])

        total_loss += loss.item()
        total_batches += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_batches, 1)


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    config,
    checkpoint_dir="model_checkpoints/decoder_postnet",
    val_loss=None,
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"decoder_postnet_epoch_{epoch:04d}.pt"

    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "train_loss": loss,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None, device=None, load_optimizer=True):
    device = torch.device(device or get_device())
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    loaded_keys = None
    skipped_keys = None
    missing_keys = None
    if load_optimizer:
        model.load_state_dict(state_dict)
    else:
        model_state = model.state_dict()
        compatible_state = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        skipped_keys = sorted(set(state_dict) - set(compatible_state))
        missing_keys = sorted(set(model_state) - set(compatible_state))
        model_state.update(compatible_state)
        model.load_state_dict(model_state)
        loaded_keys = sorted(compatible_state)

    optimizer_loaded = False
    if load_optimizer and optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer_loaded = True

    epoch = int(checkpoint.get("epoch", 0)) if isinstance(checkpoint, dict) else 0
    return {
        "checkpoint_path": checkpoint_path,
        "epoch": epoch,
        "next_epoch": epoch + 1,
        "loss": checkpoint.get("loss") if isinstance(checkpoint, dict) else None,
        "train_loss": checkpoint.get("train_loss") if isinstance(checkpoint, dict) else None,
        "val_loss": checkpoint.get("val_loss") if isinstance(checkpoint, dict) else None,
        "optimizer_loaded": optimizer_loaded,
        "loaded_keys": loaded_keys,
        "skipped_keys": skipped_keys,
        "missing_keys": missing_keys,
    }


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
    model = (model or VCPostNetDecoder()).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    dataloader = build_dataloader(dataset, training_config, shuffle=True)

    return model, optimizer, dataloader, device


def run_training(
    config=None,
    dataset=None,
    val_dataset=None,
    model=None,
    device=None,
    checkpoint_path=None,
    load_optimizer=True,
):
    config = config or load_config()
    training_config = config["training"]

    model, optimizer, train_loader, device = build_training_objects(
        config=config,
        dataset=dataset,
        model=model,
        device=device,
    )

    resume_info = None
    start_epoch = 1
    if checkpoint_path is not None:
        resume_info = load_checkpoint(
            checkpoint_path,
            model,
            optimizer=optimizer,
            device=device,
            load_optimizer=load_optimizer,
        )
        start_epoch = resume_info["next_epoch"]
        print(
            "Loaded checkpoint: "
            f"{resume_info['checkpoint_path']} "
            f"(epoch {resume_info['epoch']}, next epoch {start_epoch}, "
            f"optimizer loaded: {resume_info['optimizer_loaded']})"
        )
        if resume_info["loaded_keys"] is not None:
            print(
                "Loaded compatible model weights only: "
                f"{len(resume_info['loaded_keys'])} tensors loaded, "
                f"{len(resume_info['missing_keys'])} tensors newly initialized."
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
    val_loader = None
    if val_dataset is not None:
        val_loader = build_dataloader(val_dataset, training_config, shuffle=False)

    checkpoint_dir = training_config.get("model_checkpoint_dir") or "model_checkpoints/decoder_postnet"
    if start_epoch > training_config["num_epochs"]:
        print(
            f"Checkpoint is already at epoch {start_epoch - 1}; "
            f"num_epochs is {training_config['num_epochs']}. Nothing to train."
        )
        return model, optimizer, {"history": history, "resume_info": resume_info}

    for epoch in range(start_epoch, training_config["num_epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, device) if val_loader is not None else None
        history_item = {"epoch": epoch, "loss": train_loss, "train_loss": train_loss, "val_loss": val_loss}
        history.append(history_item)

        if val_loss is None:
            print(f"Epoch {epoch}/{training_config['num_epochs']} - train loss: {train_loss:.4f}")
        else:
            print(
                f"Epoch {epoch}/{training_config['num_epochs']} - "
                f"train loss: {train_loss:.4f} - val loss: {val_loss:.4f}"
            )

        if epoch % training_config["save_interval"] == 0:
            checkpoint_path = save_checkpoint(
                model,
                optimizer,
                epoch,
                train_loss,
                config,
                checkpoint_dir,
                val_loss=val_loss,
            )
            print(f"Saved checkpoint: {checkpoint_path}")

    return model, optimizer, {"history": history, "resume_info": resume_info}


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


class PostNet(nn.Module):
    def __init__(self, mel_dim=80, hidden_dim=512, kernel_size=5, num_layers=5, dropout=0.5):
        super().__init__()
        if num_layers < 2:
            raise ValueError("PostNet requires at least 2 convolution layers")

        layers = [
            nn.Sequential(
                ConvNorm(mel_dim, hidden_dim, kernel_size=kernel_size),
                nn.BatchNorm1d(hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
            )
        ]

        for _ in range(num_layers - 2):
            layers.append(
                nn.Sequential(
                    ConvNorm(hidden_dim, hidden_dim, kernel_size=kernel_size),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                )
            )

        layers.append(
            nn.Sequential(
                ConvNorm(hidden_dim, mel_dim, kernel_size=kernel_size),
                nn.BatchNorm1d(mel_dim),
                nn.Dropout(dropout),
            )
        )
        self.layers = nn.ModuleList(layers)

    def forward(self, mel):
        output = mel.transpose(1, 2)
        for layer in self.layers:
            output = layer(output)
        return output.transpose(1, 2)


class VCPostNetDecoder(nn.Module):
    def __init__(
        self,
        content_dim=768,
        speaker_dim=192,
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

        self.postnet = PostNet(
            mel_dim=mel_dim,
            hidden_dim=prenet_dim,
            kernel_size=conv_kernel_size,
            num_layers=5,
            dropout=dropout,
        )

    def forward(self, content_latent, speaker_latent, return_coarse=False):
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
        coarse_mel = self.linear_projection(output)
        refined_mel = coarse_mel + self.postnet(coarse_mel)
        if return_coarse:
            return coarse_mel, refined_mel
        return refined_mel


if __name__ == "__main__":
    decoder = VCPostNetDecoder()
    content = torch.randn(2, 100, 768)
    speaker = torch.randn(2, 1, 192)
    coarse_mel, mel = decoder(content, speaker, return_coarse=True)
    print("Coarse mel shape:", coarse_mel.shape)
    print("Output mel shape:", mel.shape)