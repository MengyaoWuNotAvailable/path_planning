# octomap.py

class OctoMap:
    """
    简易 OctoMap, 仅示例：
      - coarse_resolution
      - fine_resolution
    对于 coarse层: 若子体素有任何障碍，则该节点视为阻塞
    对于 fine层: 类似体素级精度
    """

    def __init__(self, env, coarse_resolution=4, fine_resolution=1):
        self.env = env
        self.coarse_res = coarse_resolution
        self.fine_res = fine_resolution

        # 记录coarse/fine各自网格的尺寸(节点数)
        import math
        self.coarse_width = (env.width  + coarse_resolution - 1)//coarse_resolution
        self.coarse_height= (env.height + coarse_resolution - 1)//coarse_resolution
        self.coarse_depth = (env.depth  + coarse_resolution - 1)//coarse_resolution

        self.fine_width = (env.width  + fine_resolution -1)//fine_resolution
        self.fine_height= (env.height + fine_resolution-1)//fine_resolution
        self.fine_depth = (env.depth  + fine_resolution -1)//fine_resolution

    def is_blocked_coarse(self, cx, cy, cz):
        """
        在 coarse网格坐标(cx,cy,cz), 对应到原始体素范围:
         x in [cx*res, (cx+1)*res), ...
        若任何子体素是障碍，则视为 blocked
        """
        base_x = cx * self.coarse_res
        base_y = cy * self.coarse_res
        base_z = cz * self.coarse_res
        for dx in range(self.coarse_res):
            for dy in range(self.coarse_res):
                for dz in range(self.coarse_res):
                    x = base_x + dx
                    y = base_y + dy
                    z = base_z + dz
                    if not self.env.in_bounds(x, y, z):
                        return True  # 越界就当阻塞(示例)
                    if self.env.is_blocked(x, y, z):
                        return True
        return False

    def is_blocked_fine(self, fx, fy, fz):
        """
        在 fine网格坐标对应到原始体素
        """
        base_x = fx * self.fine_res
        base_y = fy * self.fine_res
        base_z = fz * self.fine_res
        for dx in range(self.fine_res):
            for dy in range(self.fine_res):
                for dz in range(self.fine_res):
                    x = base_x + dx
                    y = base_y + dy
                    z = base_z + dz
                    if not self.env.in_bounds(x, y, z):
                        return True
                    if self.env.is_blocked(x, y, z):
                        return True
        return False

    def coarse_in_bounds(self, cx, cy, cz):
        return 0 <= cx < self.coarse_width and 0 <= cy < self.coarse_height and 0 <= cz < self.coarse_depth

    def fine_in_bounds(self, fx, fy, fz):
        return 0 <= fx < self.fine_width and 0 <= fy < self.fine_height and 0 <= fz < self.fine_depth
