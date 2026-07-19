import torch
import torch.nn as nn
from transformers import HubertModel


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        # Final projection is added for fairseq backward compatibility,
        # but is typically bypassed during inference to isolate pure content vectors.
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


model_id = "lengyue233/content-vec-best"


def get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_content_encoder(device=None):
    device = torch.device(device or get_default_device())
    content_encoder: HubertModelWithFinalProj = HubertModelWithFinalProj.from_pretrained(model_id)
    nn.Module.to(content_encoder, device)
    content_encoder.eval()
    return content_encoder


def encode_content(content_encoder, audio_tensor, device=None):
    if device is None:
        device = next(content_encoder.parameters()).device
    device = torch.device(device)

    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    audio_tensor = audio_tensor.to(device)

    with torch.no_grad():
        outputs = content_encoder(audio_tensor)
    return outputs["last_hidden_state"]


if __name__ == "__main__":
    device = get_default_device()
    content_encoder = load_content_encoder(device=device)
    dummy_audio = torch.randn(1, 16000, device=device)
    with torch.no_grad():
        outputs = content_encoder(dummy_audio)

    content_vectors = outputs["last_hidden_state"]
    print("Output Shape:", content_vectors.shape)