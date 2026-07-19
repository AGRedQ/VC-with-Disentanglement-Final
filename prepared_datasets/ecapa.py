import torch
from speechbrain.inference.speaker import EncoderClassifier

SAMPLING_RATE = 16000


def get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_speaker_encoder(device=None):
    device = torch.device(device or get_default_device())
    speaker_encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)},
    )
    assert speaker_encoder is not None
    return speaker_encoder

def encode_speaker(speaker_encoder, audio_tensor, device=None):
    if device is None:
        device = getattr(speaker_encoder, "device", get_default_device())
    device = torch.device(device)

    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    audio_tensor = audio_tensor.to(device)

    with torch.no_grad():
        speaker_latent = speaker_encoder.encode_batch(audio_tensor)
    return speaker_latent
if __name__ == "__main__":
    device = get_default_device()
    speaker_encoder = load_speaker_encoder(device=device)
    input_example = torch.rand(1, SAMPLING_RATE, device=device)
    speaker_latent = encode_speaker(speaker_encoder, input_example, device=device)
    print("Latent representation shape:", speaker_latent.shape)