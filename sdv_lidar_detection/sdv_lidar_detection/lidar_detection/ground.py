import numpy as np

def _fit_plane(a, b, c):
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n)
    if norm < 1e-8:
        return None
    n = n / norm
    return n, -float(n @ a)

def remove_ground(points, threshold=0.15, iterations=120, min_inliers=100,
                   max_tilt_deg=12.0, rng_seed=7):
    if len(points) < 3:
        return points, np.empty((0, 3), dtype=np.float32), None

    rng = np.random.default_rng(rng_seed)
    best = None
    cos_limit = np.cos(np.deg2rad(max_tilt_deg))

    for _ in range(iterations):
        ids = rng.choice(len(points), 3, replace=False)
        plane = _fit_plane(points[ids[0]], points[ids[1]], points[ids[2]])
        if plane is None:
            continue
        n, d = plane
        if abs(float(n[2])) < cos_limit:
            continue
        distances = np.abs(points @ n + d)
        inliers = distances <= threshold
        count = int(inliers.sum())
        if count < min_inliers:
            continue
        score = count - 0.1 * abs(float(np.median(points[inliers, 2])))
        if best is None or score > best[0]:
            best = (score, inliers, n, d)

    if best is None:
        return points, np.empty((0, 3), dtype=np.float32), None
    _, inliers, n, d = best
    return points[~inliers], points[inliers], (n, d)
