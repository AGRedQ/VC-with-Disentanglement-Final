import torch
from speechbrain.inference.speaker import EncoderClassifier

SAMPLING_RATE = 16000


def load_speaker_encoder():
    speaker_encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
    )
    assert speaker_encoder is not None
    return speaker_encoder

def encode_speaker(speaker_encoder, audio_tensor):
    speaker_latent = speaker_encoder.encode_batch(audio_tensor)
    return speaker_latent
if __name__ == "__main__":
    speaker_encoder = load_speaker_encoder()
    input_example = torch.rand(1, SAMPLING_RATE)
    speaker_latent = speaker_encoder.encode_batch(input_example)
    print("Latent representation shape:", speaker_latent.shape)