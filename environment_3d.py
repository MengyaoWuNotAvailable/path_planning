# environment_3d.py

import numpy as np

class Environment3D:
    """
    3D 体素栅格环境
    - width, height, depth: 尺寸
    - obstacles: set((x,y,z), ...) 障碍物
    """

    def __init__(self, width=20, height=20, depth=20, obstacle_positions=None):
        self.width = width
        self.height = height
        self.depth = depth
        self.obstacles = obstacle_positions if obstacle_positions else set()

    def in_bounds(self, x, y, z):
        return 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth

    def is_blocked(self, x, y, z):
        if not self.in_bounds(x, y, z):
            return True
        return (x, y, z) in self.obstacles

    def random_free_position(self):
        """在非障碍区域随机找一个体素坐标。"""
        while True:
            rx = np.random.randint(0, self.width)
            ry = np.random.randint(0, self.height)
            rz = np.random.randint(0, self.depth)
            if (rx, ry, rz) not in self.obstacles:
                return (rx, ry, rz)
