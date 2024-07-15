from Sim3DR import get_normal, rasterize
from Sim3DR.lighting import _norm, norm_vertices
from utils.render import _to_ctype
import numpy as np


class FaceCulling:
    def __init__(self, triangles):
        self.camera_pos = np.array([0, 0, 1e10], dtype=np.float32).reshape((1, -1))
        self.triangles = triangles

    def __call__(self, vertices):
        vertices = _to_ctype(vertices.T)
        normal = get_normal(vertices, self.triangles)

        # 2. lighting
        vertices_n = norm_vertices(vertices.copy())

        # diffuse component
        direction = _norm(self.camera_pos - vertices_n)
        cos = np.sum(normal * direction, axis=1)[:, None]
        visibility_mask = cos.squeeze() > 0

        return visibility_mask

    def get_normales(self,vertices):
        vertices = _to_ctype(vertices.T)
        return get_normal(vertices,self.triangles)
    
