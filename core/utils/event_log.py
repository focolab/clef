"""
EventLog — lightweight append-only event log with high-resolution timestamps.

Used by BaseInputDevice, BaseOutputDevice, and BaseClosedLoopLogic to
automatically record when actions occur during a CLEF session.
"""

import time
from typing import Any, List, Tuple


class EventLog:
    """Append-only log of (relative_timestamp, payload) tuples.

    Timestamps use ``time.perf_counter()`` for monotonic, high-resolution
    timing.  A wall-clock ``t0`` is stored once at construction so that
    relative timestamps can be mapped back to absolute time after the session.

    Disable recording by setting ``enabled = False``.
    """

    def __init__(self):
        self._t0_perf: float = time.perf_counter()
        self._t0_wall: float = time.time()
        self._events: List[Tuple[int, float, Any]] = []
        self._call_count: int = 0
        self.enabled: bool = True

    def record(self, payload: Any = None) -> float:
        """Append an event. Returns the relative timestamp, or -1.0 if disabled."""
        if not self.enabled:
            return -1.0
        t = time.perf_counter() - self._t0_perf
        self._events.append((self._call_count, t, payload))
        self._call_count += 1
        return t

    @property
    def events(self) -> List[Tuple[int, float, Any]]:
        return self._events

    @property
    def timestamps(self) -> List[float]:
        return [e[1] for e in self._events]

    @property
    def sample_indices(self) -> List[int]:
        return [e[0] for e in self._events]

    def to_dict(self) -> dict:
        return {
            "t0_wall": self._t0_wall,
            "events": list(self._events),
        }
