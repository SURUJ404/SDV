import numpy as np

def oriented_bbox(points):
    xy = points[:, :2]
    center_xy = xy.mean(axis=0)
    centered = xy - center_xy
    cov = centered.T @ centered / max(len(points) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    yaw = float(np.arctan2(axis[1], axis[0]))
    axes = np.column_stack([axis, np.array([-axis[1], axis[0]])])
    local = centered @ axes
    mins, maxs = local.min(axis=0), local.max(axis=0)
    dims_xy = maxs - mins
    center_local = (mins + maxs) * 0.5
    center_xy = center_xy + center_local @ axes.T
    zmin, zmax = float(points[:,2].min()), float(points[:,2].max())
    center = np.array([center_xy[0], center_xy[1], (zmin+zmax)/2], dtype=np.float32)
    dims = np.array([dims_xy[0], dims_xy[1], zmax-zmin], dtype=np.float32)
    return center, dims, yaw

def physical_filter(dims, limits):
    l, w, h = map(float, dims)
    return (limits["min_length_m"] <= l <= limits["max_length_m"] and
            limits["min_width_m"] <= w <= limits["max_width_m"] and
            limits["min_height_m"] <= h <= limits["max_height_m"])
