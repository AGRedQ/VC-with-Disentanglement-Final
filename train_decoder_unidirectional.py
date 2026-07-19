import argparse

import torch

from decoder_unidirectional import VCDecoder, get_device, run_training
from vc_dataset import VCDataset, load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train the unidirectional VC decoder.")
    parser.add_argument("--config", default="configs.yaml", help="Path to config YAML file.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--smoke-test", action="store_true", help="Run one smoke training pass only.")
    parser.add_argument("--device", default=None, help="Training device, for example cuda or cpu.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    training_config = config["training"]

    if args.epochs is not None:
        training_config["num_epochs"] = args.epochs
    if args.batch_size is not None:
        training_config["batch_size"] = args.batch_size
    if args.lr is not None:
        training_config["learning_rate"] = args.lr
    training_config["smoke_test_training"] = args.smoke_test

    device = torch.device(args.device or get_device())
    train_dataset = VCDataset(config=config, split="train")
    val_dataset = VCDataset(config=config, split="val")
    test_dataset = VCDataset(config=config, split="test")

    print("Unidirectional training run setup")
    print("=" * 67)
    print(f"Device:        {device}")
    print(f"CUDA visible:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device:   {torch.cuda.get_device_name(0)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")
    print()
    print("Training parameters")
    print("=" * 67)
    for key, value in training_config.items():
        print(f"{key}: {value}")

    model = VCDecoder()
    model, optimizer, training_info = run_training(
        config=config,
        dataset=train_dataset,
        model=model,
        device=device,
    )
    print(training_info)


if __name__ == "__main__":
    main()