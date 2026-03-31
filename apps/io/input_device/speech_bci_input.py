"""
SpeechBCI input device for CLEF.

Streams pre-recorded neural spike-band power data (256 channels, 20ms bins)
from the BCI speech decoding dataset. Owns shared memory for communicating
with a GRU decoder subprocess.
"""

import logging
import pickle
import re
import subprocess
import threading
import time
from multiprocessing import shared_memory
from multiprocessing.shared_memory import ShareableList
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

import numpy as np

from core.io.input_device.BaseInputDevice import BaseInputDevice

logger = logging.getLogger(__name__)

MAX_TRIAL_BINS = 1024
N_CHANNELS = 256
DECODED_TEXT_MAX_LEN = 512


def _stream_subprocess_output(pipe, log_func):
    """Read lines from a subprocess pipe and log them."""
    for line in iter(pipe.readline, b""):
        log_func(line.decode().rstrip())
    pipe.close()


class SpeechBCIInputDevice(BaseInputDevice):
    device_class: ClassVar[Optional[str]] = "speech_bci_input"
    device_type: ClassVar[Optional[str]] = "bci"

    def __init__(self, name: str, config: Dict[str, Any] | None = None, io_manager: Any = None):
        super().__init__(name=name, config=config, io_manager=io_manager)

        self.dataset_path = config.get("dataset_path", "/home/raymonddunn/data/speechBCI/ptDecoder_ctc")
        self.partition = config.get("partition", "test")
        self.day_indices = config.get("day_indices", None)
        self.playback_rate_hz = config.get("playback_rate_hz", 50)
        self.inter_trial_pause_s = config.get("inter_trial_pause_s", 3.0)
        self.decode_interval_bins = config.get("decode_interval_bins", 10)
        self.min_decode_bins = config.get("min_decode_bins", 32)
        self.decoder_load_timeout_s = config.get("decoder_load_timeout_s", 200)
        self.decoder_python = config.get("decoder_python")
        if not self.decoder_python:
            raise ValueError("decoder_python must be set in config (path to the speech env's python binary)")

        self.loaded_data = None
        self.trials = []  # list of (day_idx, trial_idx, sentenceDat, transcription)
        self.current_trial_list_idx = 0
        self.current_bin_idx = 0
        self.total_bins = 0
        self.current_transcription = ""
        self.current_day_idx = 0
        self.trial_finished = False
        self.all_trials_done = False
        self.pause_until = 0.0

        # Shared memory
        self._neural_shm = None
        self._neural_shm_meta = None
        self._decoded_shm_text = None
        self._decoded_shm_meta = None
        self._neural_buf = None

        # Subprocess
        self._decoder_proc = None
        self._last_tick_time = 0.0

    def connect(self):
        # Load dataset
        logger.info(f"Loading dataset from {self.dataset_path}")
        with open(self.dataset_path, "rb") as f:
            self.loaded_data = pickle.load(f)

        partition_data = self.loaded_data[self.partition]
        if self.day_indices is None:
            self.day_indices = list(range(len(partition_data)))

        # Build trial list
        for list_pos, day_idx in enumerate(self.day_indices):
            day_data = partition_data[list_pos]
            n_trials = len(day_data["sentenceDat"])
            for trial_idx in range(n_trials):
                sentence_dat = day_data["sentenceDat"][trial_idx]
                transcription = day_data["transcriptions"][trial_idx].strip()
                transcription = re.sub(r"[^a-zA-Z\- ']", "", transcription)
                transcription = transcription.replace("--", "").lower()
                self.trials.append((day_idx, trial_idx, sentence_dat, transcription))

        logger.info(f"Loaded {len(self.trials)} trials from partition '{self.partition}'")

        # Create shared memory
        neural_nbytes = MAX_TRIAL_BINS * N_CHANNELS * 4  # float32
        self._neural_shm = shared_memory.SharedMemory(
            name="neural_shm_buffer", create=True, size=neural_nbytes
        )
        self._neural_buf = np.ndarray(
            (MAX_TRIAL_BINS, N_CHANNELS), dtype=np.float32, buffer=self._neural_shm.buf
        )
        self._neural_buf[:] = 0

        # meta: [bin_count, trial_ready_flag, day_idx, trial_idx, decoder_ready, last_decoded_bin_count, trial_seq]
        self._neural_shm_meta = ShareableList([0, 0, 0, 0, 0, 0, 0], name="neural_shm_meta")

        # decoded text: single string slot
        self._decoded_shm_text = ShareableList([" " * DECODED_TEXT_MAX_LEN], name="decoded_shm_text")

        # decoded meta: [decoded_ready_flag, is_final]
        self._decoded_shm_meta = ShareableList([0, 0], name="decoded_shm_meta")

        # Spawn decoder subprocess
        decoder_script = str(Path(__file__).resolve().parent.parent.parent / "subprocess" / "speech_bci_decoder.py")
        cmd = [
            self.decoder_python, decoder_script,
            "--neural-shm-name", "neural_shm_buffer",
            "--neural-meta-name", "neural_shm_meta",
            "--decoded-text-name", "decoded_shm_text",
            "--decoded-meta-name", "decoded_shm_meta",
            "--decode-interval-bins", str(self.decode_interval_bins),
            "--min-decode-bins", str(self.min_decode_bins),
        ]
        logger.info(f"Spawning decoder subprocess: {' '.join(cmd)}")
        self._decoder_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

        # Stream subprocess output to logger in background threads
        threading.Thread(
            target=_stream_subprocess_output,
            args=(self._decoder_proc.stdout, logger.info),
            daemon=True,
        ).start()
        threading.Thread(
            target=_stream_subprocess_output,
            args=(self._decoder_proc.stderr, logger.warning),
            daemon=True,
        ).start()

        # Wait for decoder to signal ready
        logger.info(f"Waiting for decoder subprocess to load model (timeout={self.decoder_load_timeout_s}s)...")
        start = time.time()
        while time.time() - start < self.decoder_load_timeout_s:
            if self._decoder_proc.poll() is not None:
                raise RuntimeError(
                    f"Decoder subprocess exited with code {self._decoder_proc.returncode} during startup"
                )
            if self._neural_shm_meta[4] == 1:
                logger.info("Decoder subprocess ready")
                break
            time.sleep(0.5)
        else:
            self._decoder_proc.kill()
            raise TimeoutError(
                f"Decoder subprocess did not become ready within {self.decoder_load_timeout_s}s"
            )

        # Load first trial
        self._load_trial(0)

    def _load_trial(self, trial_list_idx: int):
        if trial_list_idx >= len(self.trials):
            self.all_trials_done = True
            logger.info("All trials completed")
            return

        self.current_trial_list_idx = trial_list_idx
        day_idx, trial_idx, sentence_dat, transcription = self.trials[trial_list_idx]
        self.current_day_idx = day_idx
        self.current_bin_idx = 0
        self.total_bins = sentence_dat.shape[0]
        self.current_transcription = transcription
        self.trial_finished = False

        # Clear shm
        self._neural_buf[:] = 0
        self._neural_shm_meta[0] = 0  # bin_count
        self._neural_shm_meta[1] = 0  # trial_ready_flag
        self._neural_shm_meta[2] = day_idx
        self._neural_shm_meta[3] = trial_idx
        self._decoded_shm_meta[0] = 0
        self._decoded_shm_meta[1] = 0
        self._neural_shm_meta[5] = 0  # last_decoded_bin_count
        self._neural_shm_meta[6] = trial_list_idx  # trial_seq

        logger.info(
            f"Trial {trial_list_idx + 1}/{len(self.trials)}: "
            f"day={day_idx}, bins={self.total_bins}, "
            f"text='{transcription[:60]}'"
        )

    def _get_input(self) -> Any:
        # Always rate-limit to prevent burning through engine iterations
        now = time.time()
        min_interval = 1.0 / self.playback_rate_hz
        elapsed = now - self._last_tick_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_tick_time = time.time()

        if self.all_trials_done:
            return None

        # Inter-trial pause
        if self.pause_until > 0:
            if time.time() < self.pause_until:
                return None
            self.pause_until = 0.0
            self._load_trial(self.current_trial_list_idx + 1)
            if self.all_trials_done:
                return None

        if self.trial_finished:
            # Start inter-trial pause
            self.pause_until = time.time() + self.inter_trial_pause_s
            return None

        # Stream one bin
        _, _, sentence_dat, _ = self.trials[self.current_trial_list_idx]
        bin_data = sentence_dat[self.current_bin_idx]  # shape (256,)
        self._neural_buf[self.current_bin_idx] = bin_data
        self.current_bin_idx += 1
        self._neural_shm_meta[0] = self.current_bin_idx  # bin_count

        if self.current_bin_idx >= self.total_bins:
            # Signal trial ready for decoding
            self._neural_shm_meta[1] = 1  # trial_ready_flag
            self.trial_finished = True
            logger.info(f"Trial ready for decoding ({self.total_bins} bins)")

        return bin_data

    def get_decoded_text(self) -> Optional[tuple[str, bool]]:
        if self._decoded_shm_meta is None:
            return None
        if self._decoded_shm_meta[0] == 1:
            text = self._decoded_shm_text[0].strip()
            is_final = self._decoded_shm_meta[1] == 1
            self._decoded_shm_meta[0] = 0
            return (text, is_final)
        return None

    def close(self):
        if self._decoder_proc is not None:
            logger.info("Terminating decoder subprocess")
            self._decoder_proc.terminate()
            try:
                self._decoder_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._decoder_proc.kill()
            self._decoder_proc = None

        for shm in [self._neural_shm]:
            if shm is not None:
                shm.close()
                shm.unlink()

        for sl in [self._neural_shm_meta, self._decoded_shm_text, self._decoded_shm_meta]:
            if sl is not None:
                sl.shm.close()
                sl.shm.unlink()

        logger.info("SpeechBCIInputDevice closed")
