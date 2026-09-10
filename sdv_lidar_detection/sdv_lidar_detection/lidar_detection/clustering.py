import numpy as np
from scipy.spatial import cKDTree

def dbscan(points, eps=0.65, min_points=8):
    n = len(points)
    if n == 0:
        return np.empty(0, dtype=np.int32)

    tree = cKDTree(points[:, :2])
    neighbors = tree.query_ball_tree(tree, eps)
    labels = np.full(n, -1, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        seeds = list(neighbors[i])
        if len(seeds) < min_points:
            continue

        labels[i] = cluster_id
        head = 0
        queued = set(seeds)
        while head < len(seeds):
            j = seeds[head]
            head += 1
            if not visited[j]:
                visited[j] = True
                jn = neighbors[j]
                if len(jn) >= min_points:
                    for k in jn:
                        if k not in queued:
                            seeds.append(k)
                            queued.add(k)
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1

    return labels

def clusters_from_labels(points, labels, max_points=20000):
    return [
        points[idx] for cid in np.unique(labels) if cid >= 0
        for idx in [np.flatnonzero(labels == cid)]
        if len(idx) <= max_points
    ]
