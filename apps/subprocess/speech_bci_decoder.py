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

from apps.subprocess.speech_bci_inference import load_model, load_lm_decoder, decode_neural_bins

logging.basicConfig(level=logging.INFO, format="[decoder] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

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

    device = "cpu"
    model = load_model(device=device)
    ngram_decoder = load_lm_decoder()

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

            X = neural_buf[:bin_count].copy()
            decoded, decode_latency_ms = decode_neural_bins(
                model, ngram_decoder, X, day_idx, device=device
            )
            logger.info(f"Decoded ({decode_latency_ms:.1f}ms): '{decoded}'")

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
