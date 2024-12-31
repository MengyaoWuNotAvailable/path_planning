import numpy as np

class Environment3D:
    """
    3D 网格环境:
     - width, height, depth
     - obstacles: set((x,y,z), ...)
    """

    def __init__(self, width=20, height=20, depth=20, obstacle_positions=None):
        self.width = width
        self.height= height
        self.depth = depth
        self.obstacles = obstacle_positions if obstacle_positions else set()

    def in_bounds(self, x, y, z):
        return (0 <= x < self.width and
                0 <= y < self.height and
                0 <= z < self.depth)

    def is_blocked(self, x, y, z):
        if not self.in_bounds(x, y, z):
            return True
        return (x, y, z) in self.obstacles

    def random_free_position(self):
        """在非障碍坐标中随机返回 (x,y,z)"""
        while True:
            rx = np.random.randint(0, self.width)
            ry = np.random.randint(0, self.height)
            rz = np.random.randint(0, self.depth)
            if (rx, ry, rz) not in self.obstacles:
                return (rx, ry, rz)
