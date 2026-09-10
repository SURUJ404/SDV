import numpy as np
import yaml

class LidarCameraCalibration:
    def __init__(self, K, R, t, distortion=None):
        self.K = np.asarray(K, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.t = np.asarray(t, dtype=np.float64).reshape(3)
        self.distortion = (np.zeros(5, dtype=np.float64) if distortion is None
                           else np.asarray(distortion))
        if self.K.shape != (3,3) or self.R.shape != (3,3):
            raise ValueError("K and R must be 3x3")
        if not np.allclose(self.R.T @ self.R, np.eye(3), atol=1e-3):
            raise ValueError("R is not orthonormal")
        if not (0.99 <= np.linalg.det(self.R) <= 1.01):
            raise ValueError("R is not a proper rotation matrix")

    @classmethod
    def from_yaml(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return cls(d["K"], d["R"], d["t"], d.get("distortion"))

    def lidar_to_camera(self, points):
        p = np.asarray(points, dtype=np.float64)
        return (self.R @ p.T).T + self.t

    def project(self, points, image_width=None, image_height=None):
        pc = self.lidar_to_camera(points)
        z = pc[:,2]
        valid = z > 1e-6
        uv = np.full((len(pc),2), np.nan, dtype=np.float64)
        uv[valid] = (pc[valid] @ self.K.T)[:,:2] / z[valid,None]
        if image_width is not None:
            valid &= (uv[:,0] >= 0) & (uv[:,0] < image_width)
        if image_height is not None:
            valid &= (uv[:,1] >= 0) & (uv[:,1] < image_height)
        return uv, valid
