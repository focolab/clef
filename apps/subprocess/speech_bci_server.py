"""
FastAPI WebSocket server for cloud-hosted speechBCI decoding.

Receives neural bins over WebSocket, accumulates them, runs GRU + n-gram
decode at configurable intervals, streams partial/final decoded text back.

Run locally:
    uvicorn apps.subprocess.speech_bci_server:app --host 0.0.0.0 --port 8765
"""

import logging
import json
import os
import time

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from apps.subprocess.speech_bci_inference import load_model, load_lm_decoder, decode_neural_bins

logging.basicConfig(level=logging.INFO, format="[server] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# Loaded once at startup
model = None
ngram_decoder = None
device = "cpu"

# In Docker, PYTHONPATH handles imports and these env vars point to baked-in model files.
# Locally, the defaults from speech_bci_inference.py are used (no env vars needed).
SERVER_MODEL_PATH = os.environ.get("BCI_MODEL_PATH")
SERVER_LM_DIR = os.environ.get("BCI_LM_DIR")

DECODE_INTERVAL_BINS = 10
MIN_DECODE_BINS = 32
MAX_TRIAL_BINS = 1024
N_CHANNELS = 256


@app.get("/health")
async def health():
    return {
        "status": "ok" if model is not None and ngram_decoder is not None else "loading",
        "model_loaded": model is not None,
        "lm_loaded": ngram_decoder is not None,
    }


@app.on_event("startup")
async def startup():
    global model, ngram_decoder
    kwargs_model = {"device": device}
    if SERVER_MODEL_PATH:
        kwargs_model["model_path"] = SERVER_MODEL_PATH
    model = load_model(**kwargs_model)

    kwargs_lm = {}
    if SERVER_LM_DIR:
        kwargs_lm["lm_dir"] = SERVER_LM_DIR
    ngram_decoder = load_lm_decoder(**kwargs_lm)
    logger.info("Server ready")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("Client connected")

    # Per-connection state
    bins = np.zeros((MAX_TRIAL_BINS, N_CHANNELS), dtype=np.float32)
    bin_count = 0
    last_decoded_bin_count = 0
    current_trial_seq = -1
    decode_interval = DECODE_INTERVAL_BINS
    min_decode_bins = MIN_DECODE_BINS

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg["type"]

            if msg_type == "bin":
                trial_seq = msg.get("trial_seq", 0)
                if trial_seq != current_trial_seq:
                    # New trial — reset accumulator
                    bins[:] = 0
                    bin_count = 0
                    last_decoded_bin_count = 0
                    current_trial_seq = trial_seq

                data = np.array(msg["data"], dtype=np.float32)
                if bin_count < MAX_TRIAL_BINS:
                    bins[bin_count] = data
                    bin_count += 1

                # Check if we should decode
                if bin_count >= min_decode_bins and (bin_count - last_decoded_bin_count) >= decode_interval:
                    day_idx = msg.get("day_idx", 0)
                    decoded, decode_latency_ms = decode_neural_bins(
                        model, ngram_decoder, bins[:bin_count].copy(), day_idx, device=device
                    )
                    last_decoded_bin_count = bin_count
                    logger.info(f"Partial decode ({decode_latency_ms:.1f}ms): '{decoded}'")
                    await ws.send_text(json.dumps({
                        "type": "decode",
                        "text": decoded,
                        "is_final": False,
                        "bins_decoded": bin_count,
                        "decode_latency_ms": round(decode_latency_ms, 2),
                    }))

            elif msg_type == "trial_complete":
                if bin_count >= min_decode_bins:
                    day_idx = msg.get("day_idx", 0)
                    decoded, decode_latency_ms = decode_neural_bins(
                        model, ngram_decoder, bins[:bin_count].copy(), day_idx, device=device
                    )
                    logger.info(f"Final decode ({decode_latency_ms:.1f}ms): '{decoded}'")
                    await ws.send_text(json.dumps({
                        "type": "decode",
                        "text": decoded,
                        "is_final": True,
                        "bins_decoded": bin_count,
                        "decode_latency_ms": round(decode_latency_ms, 2),
                    }))
                else:
                    logger.info(f"Trial too short ({bin_count} bins), skipping")

                # Reset for next trial
                bins[:] = 0
                bin_count = 0
                last_decoded_bin_count = 0

            elif msg_type == "config":
                if "decode_interval_bins" in msg:
                    decode_interval = msg["decode_interval_bins"]
                    logger.info(f"Config update: decode_interval={decode_interval}")
                if "min_decode_bins" in msg:
                    min_decode_bins = msg["min_decode_bins"]
                    logger.info(f"Config update: min_decode_bins={min_decode_bins}")

    except WebSocketDisconnect:
        logger.info("Client disconnected")
