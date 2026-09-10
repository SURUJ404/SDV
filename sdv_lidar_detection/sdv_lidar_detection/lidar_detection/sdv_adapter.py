import base64
import json
import numpy as np

def decode_telemetry_lidar(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return _validate(np.asarray(value, dtype=np.float32))
    if isinstance(value, str):
        try:
            return decode_telemetry_lidar(json.loads(value))
        except Exception:
            return None
    if not isinstance(value, dict):
        return None
    if "points" in value:
        return _validate(np.asarray(value["points"], dtype=np.float32))
    if "data" in value:
        raw = base64.b64decode(value["data"])
        dtype = np.dtype(value.get("dtype", "float32"))
        fields = int(value.get("fields", 3))
        if fields not in (3,4):
            raise ValueError("fields must be 3 or 4")
        arr = np.frombuffer(raw, dtype=dtype)
        if arr.size % fields:
            raise ValueError("binary payload size is not divisible by fields")
        return _validate(arr.reshape(-1, fields).astype(np.float32, copy=False))
    return None

def _validate(arr):
    if arr.ndim != 2 or arr.shape[1] not in (3,4):
        raise ValueError("decoded LiDAR must be Nx3 or Nx4")
    return arr[np.isfinite(arr).all(axis=1)]
