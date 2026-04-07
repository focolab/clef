"""
Metadata I/O utilities for CLEF.

Provides save_metadata for writing session metadata to JSON,
and lazy_serialize for pickle-based serialization to/from base64 strings.
"""

import codecs
import json
import logging
import math
import pickle
from pathlib import Path
from typing import Any, Union

logger = logging.getLogger(__name__)

FLOAT_PRECISION = 6


class CompactJSONEncoder(json.JSONEncoder):
    """JSON encoder that rounds floats and collapses short lists onto one line."""

    def default(self, o):
        return super().default(o)

    def encode(self, o):
        return self._encode(o, level=0)

    def _encode(self, o, level):
        indent = " " * level
        next_indent = " " * (level + 1)

        if isinstance(o, dict):
            if not o:
                return "{}"
            items = []
            for k in (sorted(o.keys()) if self.sort_keys else o.keys()):
                v = self._encode(o[k], level + 1)
                items.append(f'{next_indent}{json.dumps(k)}: {v}')
            return "{\n" + ",\n".join(items) + "\n" + indent + "}"

        if isinstance(o, (list, tuple)):
            if not o:
                return "[]"
            # Collapse short flat lists (e.g. event tuples) onto one line
            if all(isinstance(x, (int, float, bool, str, type(None))) for x in o) and len(o) <= 5:
                return "[" + ", ".join(self._encode(x, 0) for x in o) + "]"
            items = [next_indent + self._encode(x, level + 1) for x in o]
            return "[\n" + ",\n".join(items) + "\n" + indent + "]"

        if isinstance(o, float):
            if math.isfinite(o):
                return f"{round(o, FLOAT_PRECISION)}"
            return json.dumps(o)

        return json.dumps(o)


def save_metadata(path: Union[str, Path], metadata: dict) -> None:
    """Write metadata dict to a JSON file.

    Logs errors on failure rather than raising, so a failed metadata save
    does not abort the session.
    """
    path = Path(path)
    logger.debug(f"Saving metadata to {path}")
    try:
        with open(path, "w") as f:
            f.write(CompactJSONEncoder(sort_keys=True).encode(metadata))
            f.write("\n")
        logger.info(f"Metadata saved to {path}")
    except Exception as e:
        logger.critical(f"Exception during saving metadata file: {e}")
        logger.debug(f"Unsaved metadata: {metadata}")


def lazy_serialize(obj: Any, to_str: bool = True) -> Any:
    """Pickle-based serialization to/from base64 strings.

    Args:
        obj: Object to serialize (if to_str=True) or base64 string to
             deserialize (if to_str=False).
        to_str: If True, serialize obj -> base64 string.
                If False, deserialize base64 string -> object.
    """
    if to_str:
        b = pickle.dumps(obj)
        b64 = codecs.encode(b, "base64")
        return b64.decode()
    else:
        b64 = codecs.decode(obj.encode(), "base64")
        return pickle.loads(b64)
