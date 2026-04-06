"""
Shared inference logic for the speechBCI GRU decoder + n-gram LM.

Used by both the local SHM decoder (speech_bci_decoder.py) and the
cloud WebSocket server (speech_bci_server.py).
"""

import logging
import sys
import time

import numpy as np
import torch

# Add paths for neural_decoder imports.
# Locally: these point to the local repo checkouts (run via conda "speech" env).
# In Docker: PYTHONPATH is set in the Dockerfile, so these are no-ops / not reached.
sys.path.insert(0, "/home/raymonddunn/code/speechBCI/NeuralDecoder/neuralDecoder")
sys.path.insert(0, "/home/raymonddunn/code/speechBCI/NeuralDecoder")
sys.path.insert(0, "/home/raymonddunn/code/neural_seq_decoder/src")

from neural_decoder.neural_decoder_trainer import loadModel
import utils.rld_lmDecoderUtils as lmDecoderUtils

logger = logging.getLogger(__name__)

MODEL_PATH = "/home/raymonddunn/data/speechBCI/ouput_dir/speechBaseline4"
LM_DIR = "/home/raymonddunn/data/speechBCI/dryad_data/languageModel"


def load_model(model_path: str = MODEL_PATH, device: str = "cpu"):
    """Load the GRU decoder model."""
    logger.info("Loading GRU model...")
    model = loadModel(model_path, device=device)
    model.eval()
    logger.info(f"Model loaded (kernelLen={model.kernelLen}, strideLen={model.strideLen})")
    return model


def load_lm_decoder(lm_dir: str = LM_DIR, acoustic_scale: float = 0.5, nbest: int = 1, beam: int = 18):
    """Load the n-gram language model decoder."""
    logger.info("Loading n-gram LM decoder...")
    ngram_decoder = lmDecoderUtils.build_lm_decoder(
        lm_dir, acoustic_scale=acoustic_scale, nbest=nbest, beam=beam
    )
    logger.info("N-gram LM decoder loaded")
    return ngram_decoder


def decode_neural_bins(
    model,
    ngram_decoder,
    neural_data: np.ndarray,
    day_idx: int,
    device: str = "cpu",
    blank_penalty: float = None,
) -> tuple[str, float]:
    """
    Run GRU forward pass + n-gram LM decode on neural bin data.

    Args:
        model: Loaded GRU model
        ngram_decoder: Loaded n-gram LM decoder
        neural_data: (T, 256) float32 array of neural bins
        day_idx: Day index for the model's day embedding
        device: torch device
        blank_penalty: CTC blank penalty (default: log(7))

    Returns:
        (decoded_text, decode_latency_ms)
    """
    if blank_penalty is None:
        blank_penalty = np.log(7)

    bin_count = neural_data.shape[0]
    t_start = time.time()

    X_tensor = torch.tensor(neural_data, dtype=torch.float32).unsqueeze(0).to(device)
    day_tensor = torch.tensor([day_idx], dtype=torch.int64).to(device)

    with torch.no_grad():
        pred = model.forward(X_tensor, day_tensor)

    # Slice to valid output frames only
    valid_T_prime = (bin_count - 32) // 4 + 1
    logits = pred[0, :valid_T_prime].cpu().numpy()

    # Rearrange logits for LM decoder (blank token reordering)
    logits_rearranged = np.concatenate(
        [logits[:, 1:], logits[:, 0:1]], axis=-1
    )
    logits_rearranged = lmDecoderUtils.rearrange_speech_logits(
        logits_rearranged[None, :, :], has_sil=True
    )

    # N-gram LM decode
    decoded = lmDecoderUtils.lm_decode(
        ngram_decoder,
        logits_rearranged[0],
        blankPenalty=blank_penalty,
        returnNBest=False,
        rescore=False,
    )
    decoded = decoded.strip() if decoded else ""

    decode_latency_ms = (time.time() - t_start) * 1000
    return decoded, decode_latency_ms
