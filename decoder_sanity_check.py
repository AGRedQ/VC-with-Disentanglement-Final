
"""
Memorization sanity test for the VC feature decoder.

This intentionally trains on a fixed tiny set and evaluates on that same set.
It does NOT measure generalization or voice-conversion quality.

Success criterion:
- refined training reconstruction loss drops substantially;
- saved predicted mels visually resemble their targets;
- synthesized audio from saved predicted mels is recognizable.

Run from the repository root:
    python decoder_sanity_check.py --limit 16 --epochs 1000
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from decoder_w_features import VCFeatureDecoder, move_feature_batch_to_device
from model_utils import get_device, masked_l1_loss
from prepared_datasets.vc_dataset import (
    VCFeatureDataset,
    load_config,
    vc_feature_collate_fn,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_memorization(model, dataloader, device):
    model.eval()

    total_coarse_loss = 0.0
    total_refined_loss = 0.0
    total_items = 0
    saved_examples = []

    for batch in dataloader:
        batch = move_feature_batch_to_device(batch, device)

        coarse_mel, refined_mel = model(
            batch["content"],
            batch["speaker"],
            batch["prosody"],
            return_coarse=True,
        )

        coarse_loss = masked_l1_loss(coarse_mel, batch["mel"], batch["mask"])
        refined_loss = masked_l1_loss(refined_mel, batch["mel"], batch["mask"])

        total_coarse_loss += coarse_loss.item()
        total_refined_loss += refined_loss.item()
        total_items += 1

        if len(saved_examples) < 3:
            length = int(batch["lengths"][0].item())
            saved_examples.append(
                {
                    "sample_id": batch["sample_ids"][0],
                    "length": length,
                    "target_mel": batch["mel"][0, :length].cpu(),
                    "coarse_mel": coarse_mel[0, :length].cpu(),
                    "refined_mel": refined_mel[0, :length].cpu(),
                }
            )

    return {
        "coarse_loss": total_coarse_loss / max(total_items, 1),
        "refined_loss": total_refined_loss / max(total_items, 1),
        "examples": saved_examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sanity_check_outputs/feature_decoder_16_utterances"),
    )
    args = parser.parse_args()

    if args.limit < 1:
        raise ValueError("--limit must be at least 1")

    set_seed(args.seed)
    device = torch.device(get_device())
    config = load_config()

    # Keep the same fixed samples for every epoch.
    # Batch size 1 intentionally avoids padded frames contaminating Conv1D BatchNorm
    # statistics during this diagnostic test.
    dataset = VCFeatureDataset(
        config=config,
        split="train",
        limit=args.limit,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=vc_feature_collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    eval_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=vc_feature_collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # Dropout must be disabled for a capacity/memorization test.
    # If this cannot overfit, dropout is not the explanation.
    model = VCFeatureDecoder(dropout=0.0).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"Training on {len(dataset)} fixed utterances.")
    print(f"First samples: {dataset.sample_ids[:min(5, len(dataset))]}")

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_coarse_loss = 0.0
        epoch_refined_loss = 0.0
        num_batches = 0

        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in progress:
            batch = move_feature_batch_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)

            coarse_mel, refined_mel = model(
                batch["content"],
                batch["speaker"],
                batch["prosody"],
                return_coarse=True,
            )
            coarse_loss = masked_l1_loss(coarse_mel, batch["mel"], batch["mask"])
            refined_loss = masked_l1_loss(refined_mel, batch["mel"], batch["mask"])

            # Keep the same objective as decoder_w_features.py.
            loss = coarse_loss + refined_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_coarse_loss += coarse_loss.item()
            epoch_refined_loss += refined_loss.item()
            num_batches += 1
            progress.set_postfix(
                coarse=f"{coarse_loss.item():.4f}",
                refined=f"{refined_loss.item():.4f}",
            )

        train_coarse = epoch_coarse_loss / max(num_batches, 1)
        train_refined = epoch_refined_loss / max(num_batches, 1)

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            evaluation = evaluate_memorization(model, eval_dataloader, device)
            history.append(
                {
                    "epoch": epoch,
                    "train_coarse_loss": train_coarse,
                    "train_refined_loss": train_refined,
                    "eval_coarse_loss": evaluation["coarse_loss"],
                    "eval_refined_loss": evaluation["refined_loss"],
                }
            )
            print(
                f"Epoch {epoch:04d} | "
                f"train coarse={train_coarse:.6f}, "
                f"train refined={train_refined:.6f} | "
                f"same-16 eval coarse={evaluation['coarse_loss']:.6f}, "
                f"same-16 eval refined={evaluation['refined_loss']:.6f}"
            )

    final_evaluation = evaluate_memorization(model, eval_dataloader, device)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "sample_ids": dataset.sample_ids,
            "history": history,
            "final_coarse_loss": final_evaluation["coarse_loss"],
            "final_refined_loss": final_evaluation["refined_loss"],
        },
        args.output_dir / "memorization_checkpoint.pt",
    )
    torch.save(
        final_evaluation["examples"],
        args.output_dir / "mel_reconstruction_examples.pt",
    )

    print("\nFinished.")
    print(f"Final same-16 coarse L1:  {final_evaluation['coarse_loss']:.6f}")
    print(f"Final same-16 refined L1: {final_evaluation['refined_loss']:.6f}")
    print(f"Saved checkpoint: {args.output_dir / 'memorization_checkpoint.pt'}")
    print(f"Saved mel examples: {args.output_dir / 'mel_reconstruction_examples.pt'}")


if __name__ == "__main__":
    main()