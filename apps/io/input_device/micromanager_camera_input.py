"""
Micro-Manager Camera Input Device for CLEF.

Acquires uint16 frames from a Micro-Manager-controlled camera via pycromanager.
Instantiates its own pycromanager Core object.
"""

import logging
import time
import numpy as np
from typing import Any, ClassVar, Dict, Optional

from core.io.input_device.BaseInputDevice import BaseInputDevice

logger = logging.getLogger(__name__)


class MicroManagerCameraInput(BaseInputDevice):
    """Input device wrapping a Micro-Manager camera via pycromanager."""

    device_class: ClassVar[Optional[str]] = "micromanager_camera"
    device_type: ClassVar[Optional[str]] = "hardware"

    def __init__(
        self, name: str, config: Dict[str, Any] | None = None, io_manager: Any = None
    ):
        super().__init__(name, config, io_manager=io_manager)
        self.mmc = None
        self._roi = None
        self.width = 0
        self.height = 0
        self._acquiring = False
        self._strobed = False
        self._strobe_interval_s = 0.0
        self._next_snap_time = 0.0
        # Latency control + instrumentation (continuous mode)
        self._drop_to_latest = False
        self._report_timing = False
        self._dropped = 0
        self._last_grab_t = None
        self._t_accum = 0.0
        self._n_accum = 0
        self._backlog_accum = 0
        self._last_report_t = 0.0
        # Per-frame acquisition timestamp from MMCore image metadata (the true
        # sensor time, vs the host ingest time EventLog already records). Stored
        # in the EventLog payload — not a parallel record.
        self._capture_md = False
        self._md_ok = True       # cleared after a failure so we stop retrying
        self._md_probed = False  # log tags/timing once on the first frame
        self._last_acq_ms = None

    def connect(self):
        """Instantiate pycromanager Core connection."""
        from pycromanager import Core

        self.mmc = Core(convert_camel_case=False)
        logger.info(
            f"MicroManagerCameraInput '{self.name}' connected to pycromanager Core"
        )

    def configure(self):
        """Configure camera: ROI, exposure, binning, device properties, buffer."""
        cfg = self.config

        # Set device properties first (e.g. Binning)
        device_properties = cfg.get("device_properties", {})
        cam = self.mmc.getCameraDevice()
        if device_properties:
            for prop_name, prop_value in device_properties.items():
                try:
                    self.mmc.setProperty(cam, prop_name, prop_value)
                    logger.debug(f"Set {cam}.{prop_name} = {prop_value}")
                except Exception as e:
                    logger.warning(
                        f"Could not set camera property {prop_name}={prop_value}: {e}"
                    )

        # Set exposure
        # exposure_ms = cfg.get("exposure_ms")
        # if exposure_ms is not None:
        #     self.mmc.setExposure(float(exposure_ms))
        #     logger.debug(f"Set exposure to {exposure_ms} ms")
        # Set circular buffer
        # buffer_mb = cfg.get("buffer_memory_mb", 10000)
        # self.mmc.setCircularBufferMemoryFootprint(buffer_mb)

        # Set ROI if provided, otherwise query current
        roi = cfg.get("roi")
        if roi is not None:
            self.mmc.setROI(roi[0], roi[1], roi[2], roi[3])
            self._roi = tuple(roi)
        else:
            roi_obj = self.mmc.getROI()
            self._roi = (
                roi_obj.getX(),
                roi_obj.getY(),
                roi_obj.getWidth(),
                roi_obj.getHeight(),
            )

        self.width = self._roi[2]
        self.height = self._roi[3]

        # Strobed acquisition config
        self._strobed = bool(cfg.get("use_strobed_acquisition", False))
        self._strobe_interval_s = cfg.get("strobe_inter_frame_interval_ms") / 1000.0

        # Continuous-mode latency control. The camera free-runs into MM's
        # circular buffer; popNextImage is FIFO, so if the loop is slower than
        # the camera the buffer backs up and every frame we read is stale.
        # drop_to_latest discards the backlog each grab so we always process the
        # newest frame. report_timing logs loop rate + backlog depth to locate
        # bottlenecks.
        self._drop_to_latest = bool(cfg.get("drop_to_latest", False))
        self._report_timing = bool(cfg.get("report_timing", False))
        self._capture_md = bool(cfg.get("capture_frame_metadata", False))

        logger.info(
            f"MicroManagerCameraInput '{self.name}' configured: "
            f"ROI={self._roi}, exposure={self.mmc.getExposure()} ms, "
            f"strobed={self._strobed}"
        )

    def start_acquisition(self):
        """Start acquisition (continuous or strobed)."""
        if self._strobed:
            self._next_snap_time = time.perf_counter()
            logger.debug("Started strobed acquisition")
        else:
            self.mmc.stopSequenceAcquisition()
            self.mmc.clearCircularBuffer()
            self.mmc.startContinuousSequenceAcquisition(0)
            logger.debug("Started continuous acquisition")
        self._acquiring = True

    def stop_acquisition(self):
        """Stop acquisition."""
        if self._acquiring:
            if not self._strobed:
                self.mmc.stopSequenceAcquisition()
            self._acquiring = False
            logger.debug("Stopped acquisition")

    def _get_input(self) -> Optional[np.ndarray]:
        """Grab next frame. Auto-starts acquisition if not running."""
        if not self._acquiring:
            self.start_acquisition()

        if self._strobed:
            # Wait until next snap time
            now = time.perf_counter()
            delay = self._next_snap_time - now
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.001:
                logger.warning(f"Strobe: missed interval by {-delay*1000:.1f} ms")

            # Trigger exposure and grab result
            self.mmc.snapImage()
            img = self.mmc.getImage().astype(np.uint16)
            img = img.reshape((self.height, self.width))

            self._last_acq_ms = None  # host time only in strobed mode
            self._next_snap_time += self._strobe_interval_s
            return img

        # Continuous mode: spin-wait until buffer has an image
        while self.mmc.getRemainingImageCount() == 0:
            time.sleep(0.0001)  # fast poll

        backlog = self.mmc.getRemainingImageCount()
        # Drop the stale backlog so we always process the freshest frame; without
        # this the buffer fills monotonically and latency grows to seconds. Take
        # the newest with getLastImage + clear (one bridge transfer) rather than
        # popping every stale frame (one transfer each).
        if self._drop_to_latest and backlog > 1:
            raw, self._last_acq_ms = self._pull_frame("last")
            self.mmc.clearCircularBuffer()
            self._dropped += backlog - 1
        else:
            raw, self._last_acq_ms = self._pull_frame("next")
        img = raw.astype(np.uint16).reshape((self.height, self.width))

        if self._report_timing:
            self._report_grab(backlog)
        return img

    def get_input(self):
        """Pull a frame, then record host time + camera acquisition time.

        Overrides the base so the EventLog entry carries the MMCore acquisition
        timestamp (payload) alongside the host-side perf_counter time.
        """
        result = self._get_input()
        self.event_log.record(payload=self._last_acq_ms)
        return result

    def _pull_frame(self, which):
        """Return (raw_pixels, acq_ms).

        acq_ms is the camera/MMCore acquisition timestamp in ms, or None when
        metadata capture is off/unavailable. Pixel retrieval falls back to the
        fast plain path (getLastImage/popNextImage) on any metadata error, so
        acquisition never breaks.
        """
        if self._capture_md and self._md_ok:
            try:
                return self._pull_frame_md(which)
            except Exception as e:
                self._md_ok = False
                logger.warning(
                    f"Frame-metadata capture failed ({e}); falling back to "
                    "host timestamps (still recorded in the event log)."
                )
        raw = self.mmc.getLastImage() if which == "last" else self.mmc.popNextImage()
        return raw, None

    def _pull_frame_md(self, which):
        """Pull pixels + acquisition timestamp via the tagged-image API.

        On the first frame, log the available tag keys and the pull time so the
        correct timestamp tag and the bridge cost are both visible on the rig.
        """
        method = "getLastTaggedImage" if which == "last" else "popNextTaggedImage"
        t0 = time.perf_counter()
        ti = getattr(self.mmc, method)()
        raw = np.asarray(ti.pix)
        acq_ms = self._extract_acq_ms(ti.tags)
        if not self._md_probed:
            self._md_probed = True
            dt_ms = (time.perf_counter() - t0) * 1000.0
            keys = list(ti.tags.keys()) if hasattr(ti.tags, "keys") else "n/a"
            logger.info(
                f"[frame metadata] {method}: pull={dt_ms:.1f} ms, "
                f"acq_ms={acq_ms}, tags={keys}"
            )
            if dt_ms > 15.0:
                logger.warning(
                    "[frame metadata] pull is slow — the tagged-image bridge "
                    "may transfer pixels inefficiently. Set "
                    "capture_frame_metadata: false if it throttles the loop."
                )
        return raw, acq_ms

    def _extract_acq_ms(self, tags):
        """Best-effort numeric acquisition timestamp (ms) from image metadata."""
        for key in ("ElapsedTime-ms", "ElapsedTimems", "TimeStampMsec",
                    "Camera-TimeStamp"):
            try:
                v = tags.get(key) if hasattr(tags, "get") else tags[key]
            except Exception:
                v = None
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return None

    def _report_grab(self, backlog):
        """Accumulate loop-rate / backlog stats and log ~once per second."""
        now = time.perf_counter()
        if self._last_grab_t is not None:
            self._t_accum += now - self._last_grab_t
            self._n_accum += 1
            self._backlog_accum += backlog
        self._last_grab_t = now

        if now - self._last_report_t >= 1.0 and self._n_accum > 0:
            fps = self._n_accum / self._t_accum if self._t_accum > 0 else 0.0
            avg_backlog = self._backlog_accum / self._n_accum
            logger.info(
                f"[camera timing] loop={fps:.1f} fps  "
                f"avg backlog={avg_backlog:.1f} frames  "
                f"dropped(total)={self._dropped}  frame={self.width}x{self.height}"
            )
            self._t_accum = 0.0
            self._n_accum = 0
            self._backlog_accum = 0
            self._last_report_t = now

    @property
    def last_acquisition_ms(self) -> Optional[float]:
        """MMCore acquisition timestamp (ms) of the most recent frame.

        None when metadata capture is off, unavailable, or in strobed mode. The
        data interface publishes this alongside the frame so subprocess logic can
        time its control loop off the camera rather than the host loop.
        """
        return self._last_acq_ms

    def get_roi(self) -> tuple:
        """Return current ROI as (x, y, width, height)."""
        return self._roi

    def clear_buffer(self):
        """Clear the circular buffer."""
        self.mmc.clearCircularBuffer()

    def close(self):
        """Stop acquisition and clean up."""
        self.stop_acquisition()
        logger.info(f"MicroManagerCameraInput '{self.name}' closed")
