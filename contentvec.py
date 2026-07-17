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


def load_content_encoder():
    content_encoder = HubertModelWithFinalProj.from_pretrained(model_id)
    content_encoder.eval()
    return content_encoder


def encode_content(content_encoder, audio_tensor):
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = content_encoder(audio_tensor)
    return outputs["last_hidden_state"]


if __name__ == "__main__":
    content_encoder = load_content_encoder()
    dummy_audio = torch.randn(1, 16000)
    with torch.no_grad():
        outputs = content_encoder(dummy_audio)

    content_vectors = outputs["last_hidden_state"]
    print("Output Shape:", content_vectors.shape)