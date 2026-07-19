import torch

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch_to_device(batch, device):
    return {
        "sample_ids": batch["sample_ids"],
        "content": batch["content"].to(device),
        "speaker": batch["speaker"].to(device),
        "mel": batch["mel"].to(device),
        "lengths": batch["lengths"].to(device),
        "mask": batch["mask"].to(device),
    }

def masked_l1_loss(pred_mel, target_mel, mask):
    mask = mask.to(pred_mel.device).unsqueeze(-1)
    target_mel = target_mel.to(pred_mel.device)
    loss = (pred_mel - target_mel).abs() * mask
    denominator = (mask.sum() * pred_mel.size(-1)).clamp_min(1.0)
    return loss.sum() / denominator

