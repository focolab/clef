"""
GRU decoder subprocess for BCI speech decoding.

Runs in the 'speech' conda environment. Polls shared memory for trial-ready
signals, runs the GRU model + n-gram LM decode, writes decoded text back.
"""

import argparse
import logging
import sys
import time
from multiprocessing import shared_memory
from multiprocessing.shared_memory import ShareableList

import numpy as np
import torch

# Add paths for neural_decoder imports
sys.path.insert(0, "/home/raymonddunn/code/speechBCI/NeuralDecoder/neuralDecoder")
sys.path.insert(0, "/home/raymonddunn/code/speechBCI/NeuralDecoder")
sys.path.insert(0, "/home/raymonddunn/code/neural_seq_decoder/src")

from neural_decoder.neural_decoder_trainer import loadModel
import utils.rld_lmDecoderUtils as lmDecoderUtils

logging.basicConfig(level=logging.INFO, format="[decoder] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

MODEL_PATH = "/home/raymonddunn/data/speechBCI/ouput_dir/speechBaseline4"
LM_DIR = "/home/raymonddunn/data/speechBCI/dryad_data/languageModel"
MAX_TRIAL_BINS = 1024
N_CHANNELS = 256
DECODED_TEXT_MAX_LEN = 512


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neural-shm-name", default="neural_shm_buffer")
    parser.add_argument("--neural-meta-name", default="neural_shm_meta")
    parser.add_argument("--decoded-text-name", default="decoded_shm_text")
    parser.add_argument("--decoded-meta-name", default="decoded_shm_meta")
    parser.add_argument("--decode-interval-bins", type=int, default=10)
    parser.add_argument("--min-decode-bins", type=int, default=32)
    args = parser.parse_args()

    # Load model
    logger.info("Loading GRU model...")
    device = "cpu"
    model = loadModel(MODEL_PATH, device=device)
    model.eval()
    logger.info(f"Model loaded (kernelLen={model.kernelLen}, strideLen={model.strideLen})")

    # Load n-gram LM decoder
    logger.info("Loading n-gram LM decoder...")
    ngram_decoder = lmDecoderUtils.build_lm_decoder(
        LM_DIR, acoustic_scale=0.5, nbest=1, beam=18
    )
    logger.info("N-gram LM decoder loaded")

    blank_penalty = np.log(7)

    # Attach to shared memory
    neural_shm = shared_memory.SharedMemory(name=args.neural_shm_name, create=False)
    neural_buf = np.ndarray((MAX_TRIAL_BINS, N_CHANNELS), dtype=np.float32, buffer=neural_shm.buf)
    neural_meta = ShareableList(name=args.neural_meta_name)
    decoded_text = ShareableList(name=args.decoded_text_name)
    decoded_meta = ShareableList(name=args.decoded_meta_name)

    # Signal ready to main process
    neural_meta[4] = 1
    logger.info("Decoder ready. Polling for trials...")

    last_decoded_bin_count = 0
    last_trial_seq = -1

    try:
        while True:
            # Detect new trial via sequence counter
            trial_seq = neural_meta[6]
            if trial_seq != last_trial_seq:
                last_decoded_bin_count = 0
                last_trial_seq = trial_seq

            bin_count = neural_meta[0]
            trial_complete = neural_meta[1] == 1

            should_decode = False
            if bin_count < args.min_decode_bins:
                if trial_complete:
                    logger.info(f"Trial too short ({bin_count} bins < {args.min_decode_bins}), skipping")
                    neural_meta[1] = 0
                time.sleep(0.005)
                continue
            elif trial_complete and bin_count > last_decoded_bin_count:
                should_decode = True
            elif (bin_count - last_decoded_bin_count) >= args.decode_interval_bins:
                should_decode = True

            if not should_decode:
                time.sleep(0.005)
                continue

            day_idx = neural_meta[2]
            logger.info(f"{'Final' if trial_complete else 'Partial'} decode: {bin_count} bins, day_idx={day_idx}")

            # Read neural data
            X = neural_buf[:bin_count].copy()  # (T, 256)
            X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
            day_tensor = torch.tensor([day_idx], dtype=torch.int64).to(device)

            # Forward pass
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
            logger.info(f"Decoded: '{decoded}'")

            # Write result to shared memory
            padded = decoded.ljust(DECODED_TEXT_MAX_LEN)[:DECODED_TEXT_MAX_LEN]
            decoded_text[0] = padded
            decoded_meta[0] = 1
            decoded_meta[1] = 1 if trial_complete else 0
            neural_meta[5] = bin_count
            last_decoded_bin_count = bin_count

            if trial_complete:
                neural_meta[1] = 0

    except KeyboardInterrupt:
        logger.info("Decoder subprocess interrupted")
    finally:
        neural_shm.close()
        neural_meta.shm.close()
        decoded_text.shm.close()
        decoded_meta.shm.close()


if __name__ == "__main__":
    main()
