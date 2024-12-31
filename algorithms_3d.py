# algorithms_3d.py

import math, random, heapq
import numpy as np
from collections import deque
from octomap import OctoMap

################################################################################
# BFS3D, AStar3D, PotentialField3D, RRT3D 与之前相同，为节省篇幅，不在此重复贴长代码
# 此处我们只贴与RRT*、CoarseToFine相关的新类
# 你可以把之前 BFS3D, AStar3D, PotentialField3D, RRT3D 的实现原封不动放在本文件开头
################################################################################

# ================== 先把原先 BFS3D ... RRT3D 同样放进来 =====================

class BFS3D:
    def __init__(self, env, start, goal, neighbor_mode="6", use_octomap=None, coarse_layer=False, fine_layer=False):
        """
        新增: use_octomap 为OctoMap对象或None
             coarse_layer=True: 在octomap的coarse层检测 is_blocked_coarse
             fine_layer=True:   在octomap的fine层检测 is_blocked_fine
        """
        self.env = env
        self.start = start
        self.goal = goal
        self.neighbor_mode = neighbor_mode
        self.octomap = use_octomap
        self.coarse_layer = coarse_layer
        self.fine_layer   = fine_layer

        self.came_from = {}
        self.steps = 0

    def is_blocked(self, x, y, z):
        """根据是否use_octomap, coarse_layer/fine_layer来选择检测函数"""
        if self.octomap is None:
            # 普通 environment
            return self.env.is_blocked(x, y, z)
        else:
            if self.coarse_layer:
                return (not self.octomap.coarse_in_bounds(x,y,z)) or self.octomap.is_blocked_coarse(x,y,z)
            elif self.fine_layer:
                return (not self.octomap.fine_in_bounds(x,y,z)) or self.octomap.is_blocked_fine(x,y,z)
            else:
                return self.env.is_blocked(x,y,z)

    def in_bounds(self, x, y, z):
        if self.octomap is None:
            return self.env.in_bounds(x,y,z)
        else:
            if self.coarse_layer:
                return self.octomap.coarse_in_bounds(x,y,z)
            elif self.fine_layer:
                return self.octomap.fine_in_bounds(x,y,z)
            else:
                return self.env.in_bounds(x,y,z)

    def plan(self):
        # BFS
        if self.is_blocked(*self.start) or self.is_blocked(*self.goal):
            return None

        frontier = deque()
        frontier.append(self.start)
        self.came_from[self.start] = None

        while frontier:
            self.steps += 1
            current = frontier.popleft()
            if current == self.goal:
                return self._reconstruct_path(current)

            for nbr in self.get_neighbors(*current):
                if nbr not in self.came_from:
                    self.came_from[nbr] = current
                    frontier.append(nbr)
        return None

    def get_neighbors(self, x, y, z):
        if self.neighbor_mode == "6":
            directions = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
        else:
            directions = []
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    for dz in [-1,0,1]:
                        if dx==0 and dy==0 and dz==0:
                            continue
                        directions.append((dx,dy,dz))

        for dx, dy, dz in directions:
            nx, ny, nz = x+dx, y+dy, z+dz
            if self.in_bounds(nx, ny, nz) and not self.is_blocked(nx, ny, nz):
                yield (nx, ny, nz)

    def _reconstruct_path(self, end_pos):
        path = []
        cur = end_pos
        while cur is not None:
            path.append(cur)
            cur = self.came_from[cur]
        path.reverse()
        return path

class AStar3D:
    def __init__(self, env, start, goal, neighbor_mode="6", heuristic="euclidean",
                 use_octomap=None, coarse_layer=False, fine_layer=False):
        self.env = env
        self.start = start
        self.goal = goal
        self.neighbor_mode = neighbor_mode
        self.heuristic_type = heuristic
        self.octomap = use_octomap
        self.coarse_layer = coarse_layer
        self.fine_layer   = fine_layer

        self.came_from = {}
        self.gscore = {}
        self.fscore = {}
        self.steps = 0

    def is_blocked(self, x, y, z):
        if self.octomap is None:
            return self.env.is_blocked(x,y,z)
        else:
            if self.coarse_layer:
                return (not self.octomap.coarse_in_bounds(x,y,z)) or self.octomap.is_blocked_coarse(x,y,z)
            elif self.fine_layer:
                return (not self.octomap.fine_in_bounds(x,y,z)) or self.octomap.is_blocked_fine(x,y,z)
            else:
                return self.env.is_blocked(x,y,z)

    def in_bounds(self, x, y, z):
        if self.octomap is None:
            return self.env.in_bounds(x,y,z)
        else:
            if self.coarse_layer:
                return self.octomap.coarse_in_bounds(x,y,z)
            elif self.fine_layer:
                return self.octomap.fine_in_bounds(x,y,z)
            else:
                return self.env.in_bounds(x,y,z)

    def plan(self):
        if self.is_blocked(*self.start) or self.is_blocked(*self.goal):
            return None

        self.gscore[self.start] = 0
        self.fscore[self.start] = self.heuristic(self.start, self.goal)

        open_list = []
        heapq.heappush(open_list, (self.fscore[self.start], self.start))
        self.came_from[self.start] = None

        while open_list:
            self.steps += 1
            _, current = heapq.heappop(open_list)
            if current == self.goal:
                return self._reconstruct_path(current)

            for nbr in self.get_neighbors(*current):
                tentative_g = self.gscore[current] + 1
                if (nbr not in self.gscore) or (tentative_g < self.gscore[nbr]):
                    self.gscore[nbr] = tentative_g
                    self.fscore[nbr] = tentative_g + self.heuristic(nbr, self.goal)
                    self.came_from[nbr] = current
                    heapq.heappush(open_list, (self.fscore[nbr], nbr))

        return None

    def get_neighbors(self, x, y, z):
        if self.neighbor_mode == "6":
            directions = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
        else:
            directions = []
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    for dz in [-1,0,1]:
                        if dx==0 and dy==0 and dz==0:
                            continue
                        directions.append((dx,dy,dz))

        for dx, dy, dz in directions:
            nx, ny, nz = x+dx, y+dy, z+dz
            if self.in_bounds(nx, ny, nz) and not self.is_blocked(nx, ny, nz):
                yield (nx, ny, nz)

    def heuristic(self, pos, goal):
        x1,y1,z1 = pos
        x2,y2,z2 = goal
        if self.heuristic_type == "euclidean":
            return math.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)
        else:
            return abs(x1-x2)+abs(y1-y2)+abs(z1-z2)

    def _reconstruct_path(self, end_pos):
        path = []
        cur = end_pos
        while cur is not None:
            path.append(cur)
            cur = self.came_from[cur]
        path.reverse()
        return path

class PotentialField3D:
    def __init__(self, env, start, goal, max_steps=2000, use_octomap=None):
        self.env = env
        self.start = start
        self.goal = goal
        self.max_steps = max_steps
        self.octomap = use_octomap
        # 这里为了简化, 不做 coarse/fine. 你也可仿照 BFS 加
        self.steps = 0
        self.field = self.compute_field()

    def compute_field(self):
        W,H,D = self.env.width, self.env.height, self.env.depth
        field = np.zeros((D,H,W), dtype=float)
        gx,gy,gz = self.goal
        for z in range(D):
            for y in range(H):
                for x in range(W):
                    dist = math.sqrt((x-gx)**2+(y-gy)**2+(z-gz)**2)
                    field[z,y,x] = -dist
        # 障碍设为很大
        for (ox,oy,oz) in self.env.obstacles:
            if 0<=ox<W and 0<=oy<H and 0<=oz<D:
                field[oz,oy,ox] = 999999
        return field

    def is_blocked(self, x, y, z):
        # 不演示octomap, 直接env
        return self.env.is_blocked(x,y,z)

    def in_bounds(self, x,y,z):
        return self.env.in_bounds(x,y,z)

    def plan(self):
        if self.is_blocked(*self.start) or self.is_blocked(*self.goal):
            return None
        path = [self.start]
        current = self.start
        for _ in range(self.max_steps):
            self.steps += 1
            if current == self.goal:
                return path
            nxt = self.best_neighbor(current)
            if not nxt or nxt==current:
                return None
            path.append(nxt)
            current = nxt
        return None

    def best_neighbor(self, pos):
        (x,y,z) = pos
        candidates = [(x,y,z)]
        for (dx,dy,dz) in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            nx,ny,nz = x+dx,y+dy,z+dz
            if self.in_bounds(nx,ny,nz) and not self.is_blocked(nx,ny,nz):
                candidates.append((nx,ny,nz))
        def val(px,py,pz):
            return self.field[pz,py,px]
        best = min(candidates, key=lambda c: val(*c))
        return best if best!=pos else None

# ------------------- RRT3D -----------------------

class Node3D:
    def __init__(self, x,y,z, parent=None):
        self.x = x
        self.y = y
        self.z = z
        self.parent = parent

def distance_3d(n1, n2):
    return math.sqrt((n1.x-n2.x)**2+(n1.y-n2.y)**2+(n1.z-n2.z)**2)

class RRT3D:
    def __init__(self, env, start, goal, step_size=2, max_iter=1000, goal_threshold=2.0,
                 use_octomap=None):
        self.env = env
        self.start = Node3D(*start)
        self.goal  = Node3D(*goal)
        self.step_size = step_size
        self.max_iter  = max_iter
        self.goal_threshold = goal_threshold
        self.octomap = use_octomap

        self.node_list = [self.start]
        self.steps = 0

    def is_blocked(self, x,y,z):
        if not self.env.in_bounds(x,y,z):
            return True
        return self.env.is_blocked(x,y,z)

    def plan(self):
        if self.is_blocked(self.start.x,self.start.y,self.start.z) or \
           self.is_blocked(self.goal.x,self.goal.y,self.goal.z):
            return None

        for _ in range(self.max_iter):
            self.steps += 1
            rx = random.randint(0,self.env.width-1)
            ry = random.randint(0,self.env.height-1)
            rz = random.randint(0,self.env.depth-1)
            if self.is_blocked(rx,ry,rz):
                continue
            rnd_node = Node3D(rx,ry,rz)

            nearest = self.get_nearest_node(rnd_node)
            new_node = self.steer(nearest, rnd_node)
            if not self.is_blocked(new_node.x,new_node.y,new_node.z):
                new_node.parent = nearest
                self.node_list.append(new_node)
                if distance_3d(new_node,self.goal)<self.goal_threshold:
                    return self.build_path(new_node)
        return None

    def get_nearest_node(self, node):
        nearest = None
        min_dist= float('inf')
        for nd in self.node_list:
            d = distance_3d(nd,node)
            if d<min_dist:
                min_dist=d
                nearest=nd
        return nearest

    def steer(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dz = to_node.z - from_node.z
        dist= math.sqrt(dx*dx+dy*dy+dz*dz)
        if dist==0:
            return Node3D(from_node.x, from_node.y, from_node.z, from_node)
        scale= self.step_size/dist
        nx= from_node.x + int(round(dx*scale))
        ny= from_node.y + int(round(dy*scale))
        nz= from_node.z + int(round(dz*scale))
        return Node3D(nx,ny,nz)

    def build_path(self, node):
        path=[]
        while node is not None:
            path.append((node.x,node.y,node.z))
            node=node.parent
        path.reverse()
        return path

# ------------------- RRT*3D ----------------------

class RRTStar3D(RRT3D):
    """
    简化版 RRT*, 在扩展成功后, 在某个半径内尝试 rewire
    """
    def __init__(self, env, start, goal, step_size=2, max_iter=1000, goal_threshold=2.0,
                 rewire_radius=3.0):
        super().__init__(env, start, goal, step_size, max_iter, goal_threshold)
        self.rewire_radius = rewire_radius

    def plan(self):
        if self.is_blocked(self.start.x,self.start.y,self.start.z) or \
           self.is_blocked(self.goal.x,self.goal.y,self.goal.z):
            return None

        cost_map = {self.start: 0.0}

        for _ in range(self.max_iter):
            self.steps += 1
            rx = random.randint(0,self.env.width-1)
            ry = random.randint(0,self.env.height-1)
            rz = random.randint(0,self.env.depth-1)
            if self.is_blocked(rx,ry,rz):
                continue
            rnd_node= Node3D(rx,ry,rz)

            nearest= self.get_nearest_node(rnd_node)
            new_node= self.steer(nearest, rnd_node)
            if not self.is_blocked(new_node.x,new_node.y,new_node.z):
                new_node.parent= nearest
                cost_map[new_node]= cost_map[nearest]+distance_3d(new_node,nearest)
                self.node_list.append(new_node)

                # rewire
                for nd in self.node_list:
                    if nd==new_node or nd==nearest:
                        continue
                    if distance_3d(nd,new_node)<self.rewire_radius:
                        new_cost= cost_map[new_node]+distance_3d(new_node,nd)
                        if new_cost< cost_map.get(nd,1e9):
                            # rewire
                            nd.parent= new_node
                            cost_map[nd]= new_cost

                if distance_3d(new_node,self.goal)< self.goal_threshold:
                    return self.build_path(new_node)
        return None


# ------------------- CoarseToFine -----------------
class CoarseToFinePlanner:
    """
    简单示例:
      1) 在coarse层(OctoMap)用 BFS 找到一条粗路径
      2) 对粗路径每段, 在fine层再用A*精细化
      3) 拼接生成完整路径
    """
    def __init__(self, env, start, goal, neighbor_mode="6", refine_method="astar", octomap=None):
        self.env = env
        self.start= start
        self.goal = goal
        self.neighbor_mode= neighbor_mode
        self.refine_method= refine_method
        self.octomap= octomap
        self.steps= 0

    def plan(self):
        if not self.octomap:
            return None

        # 1) BFS 在coarse层
        BFS_coarse = BFS3D(
            env=self.env,
            start=self.map_to_coarse(self.start),
            goal=self.map_to_coarse(self.goal),
            neighbor_mode=self.neighbor_mode,
            use_octomap=self.octomap,
            coarse_layer=True
        )
        coarse_path = BFS_coarse.plan()
        self.steps += BFS_coarse.steps

        if not coarse_path:
            return None

        # 2) 对coarse路径的每相邻节点, 在fine层做 refine
        full_path = []
        for idx in range(len(coarse_path)-1):
            cA = coarse_path[idx]
            cB = coarse_path[idx+1]

            # cA -> cB 需要映射回 原始坐标
            # 这里做法: 取各自在coarse栅格中心点(或 corner?), 再用 fine上的 A*
            Apos = self.coarse_to_center(cA)
            Bpos = self.coarse_to_center(cB)

            sub_planner = None
            if self.refine_method=="astar":
                sub_planner = AStar3D(
                    env=self.env,
                    start=Apos, goal=Bpos,
                    neighbor_mode=self.neighbor_mode,
                    heuristic="euclidean",
                    use_octomap=self.octomap,
                    fine_layer=True
                )
            else:
                # BFS fine
                sub_planner = BFS3D(
                    env=self.env,
                    start=Apos, goal=Bpos,
                    neighbor_mode=self.neighbor_mode,
                    use_octomap=self.octomap,
                    fine_layer=True
                )

            sub_path = sub_planner.plan()
            self.steps += sub_planner.steps
            if not sub_path:
                return None
            if idx>0:
                sub_path = sub_path[1:]  # 避免重复拼接
            full_path.extend(sub_path)

        return full_path

    def map_to_coarse(self, pos):
        """将原始体素坐标pos -> coarse层坐标 (cx,cy,cz)"""
        (x,y,z)= pos
        cx= x//self.octomap.coarse_res
        cy= y//self.octomap.coarse_res
        cz= z//self.octomap.coarse_res
        return (cx,cy,cz)

    def coarse_to_center(self, coarse_pos):
        """将coarse节点坐标 -> 原始坐标, 这里简单返回该块体素中心"""
        (cx,cy,cz)= coarse_pos
        base_x= cx*self.octomap.coarse_res
        base_y= cy*self.octomap.coarse_res
        base_z= cz*self.octomap.coarse_res
        # center
        half= self.octomap.coarse_res//2
        rx= base_x+ half
        ry= base_y+ half
        rz= base_z+ half
        # clip到边界
        rx= min(rx,self.env.width-1)
        ry= min(ry,self.env.height-1)
        rz= min(rz,self.env.depth-1)
        return (rx,ry,rz)
