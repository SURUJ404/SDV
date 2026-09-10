import numpy as np

def validate_points(points):
    p = np.asarray(points)
    if p.ndim != 2 or p.shape[1] not in (3, 4):
        raise ValueError("point cloud must have shape Nx3 or Nx4")
    p = p[:, :3].astype(np.float32, copy=False)
    if p.size == 0:
        return p
    return p[np.isfinite(p).all(axis=1)]

def range_filter(points, min_range, max_range, min_z, max_z):
    if len(points) == 0:
        return points
    r2 = points[:, 0] ** 2 + points[:, 1] ** 2
    mask = ((r2 >= min_range ** 2) & (r2 <= max_range ** 2) &
            (points[:, 2] >= min_z) & (points[:, 2] <= max_z))
    return points[mask]

def voxel_downsample(points, voxel_size):
    if len(points) == 0 or voxel_size <= 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(unique_idx)]
