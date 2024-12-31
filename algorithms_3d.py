import math, random, heapq
import numpy as np
from collections import deque

from octomap import OctoMap

###############################################################################
# 包含 BFS3D, AStar3D, PotentialField3D, RRT3D, RRTStar3D, CoarseToFinePlanner
# 在 RRT/RRT* 中引入 goal_bias；PotentialField3D 引入 jump_when_stuck
###############################################################################


# ---------------- BFS3D ----------------
class BFS3D:
    def __init__(self, env, start, goal, neighbor_mode="6",
                 use_octomap=None, coarse_layer=False, fine_layer=False):
        self.env= env
        self.start= start
        self.goal= goal
        self.neighbor_mode= neighbor_mode
        self.octomap= use_octomap
        self.coarse_layer= coarse_layer
        self.fine_layer  = fine_layer

        self.came_from= {}
        self.steps=0

    def is_blocked(self, x,y,z):
        if not self.octomap:
            return self.env.is_blocked(x,y,z)
        else:
            if self.coarse_layer:
                if not self.octomap.coarse_in_bounds(x,y,z):
                    return True
                return self.octomap.is_blocked_coarse(x,y,z)
            elif self.fine_layer:
                if not self.octomap.fine_in_bounds(x,y,z):
                    return True
                return self.octomap.is_blocked_fine(x,y,z)
            else:
                return self.env.is_blocked(x,y,z)

    def in_bounds(self, x,y,z):
        if not self.octomap:
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

        frontier= deque()
        frontier.append(self.start)
        self.came_from[self.start]= None

        while frontier:
            self.steps+=1
            cur= frontier.popleft()
            if cur== self.goal:
                return self._reconstruct_path(cur)
            for nbr in self.get_neighbors(*cur):
                if nbr not in self.came_from:
                    self.came_from[nbr]= cur
                    frontier.append(nbr)
        return None

    def get_neighbors(self, x,y,z):
        if self.neighbor_mode=="6":
            dirs=[(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
        else:
            dirs=[]
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    for dz in [-1,0,1]:
                        if dx==0 and dy==0 and dz==0:
                            continue
                        dirs.append((dx,dy,dz))

        for (dx,dy,dz) in dirs:
            nx,ny,nz= x+dx,y+dy,z+dz
            if self.in_bounds(nx,ny,nz) and not self.is_blocked(nx,ny,nz):
                yield (nx,ny,nz)

    def _reconstruct_path(self, endpos):
        path=[]
        cur= endpos
        while cur is not None:
            path.append(cur)
            cur= self.came_from[cur]
        path.reverse()
        return path


# ---------------- AStar3D ----------------
class AStar3D:
    def __init__(self, env, start, goal, neighbor_mode="6", heuristic="euclidean",
                 use_octomap=None, coarse_layer=False, fine_layer=False):
        self.env= env
        self.start= start
        self.goal= goal
        self.neighbor_mode= neighbor_mode
        self.heuristic_type= heuristic
        self.octomap= use_octomap
        self.coarse_layer= coarse_layer
        self.fine_layer  = fine_layer

        self.came_from= {}
        self.gscore= {}
        self.fscore= {}
        self.steps=0

    def is_blocked(self, x,y,z):
        if not self.octomap:
            return self.env.is_blocked(x,y,z)
        else:
            if self.coarse_layer:
                if not self.octomap.coarse_in_bounds(x,y,z):
                    return True
                return self.octomap.is_blocked_coarse(x,y,z)
            elif self.fine_layer:
                if not self.octomap.fine_in_bounds(x,y,z):
                    return True
                return self.octomap.is_blocked_fine(x,y,z)
            else:
                return self.env.is_blocked(x,y,z)

    def in_bounds(self, x,y,z):
        if not self.octomap:
            return self.env.in_bounds(x,y,z)
        else:
            if self.coarse_layer:
                return self.octomap.coarse_in_bounds(x,y,z)
            elif self.fine_layer:
                return self.octomap.fine_in_bounds(x,y,z)
            else:
                return self.env.in_bounds(x,y,z)

    def heuristic(self, a, b):
        x1,y1,z1= a
        x2,y2,z2= b
        if self.heuristic_type=="euclidean":
            return math.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)
        else:
            return abs(x1-x2)+abs(y1-y2)+abs(z1-z2)

    def plan(self):
        if self.is_blocked(*self.start) or self.is_blocked(*self.goal):
            return None

        startf= self.heuristic(self.start, self.goal)
        self.gscore[self.start]= 0
        self.fscore[self.start]= startf
        openlist=[]
        heapq.heappush(openlist, (startf, self.start))
        self.came_from[self.start]= None

        while openlist:
            self.steps+=1
            _, cur= heapq.heappop(openlist)
            if cur== self.goal:
                return self._reconstruct_path(cur)
            for nbr in self.get_neighbors(*cur):
                tg= self.gscore[cur]+1
                if (nbr not in self.gscore) or (tg< self.gscore[nbr]):
                    self.gscore[nbr]= tg
                    self.fscore[nbr]= tg+ self.heuristic(nbr, self.goal)
                    self.came_from[nbr]= cur
                    heapq.heappush(openlist,(self.fscore[nbr], nbr))
        return None

    def get_neighbors(self, x,y,z):
        if self.neighbor_mode=="6":
            dirs=[(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
        else:
            dirs=[]
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    for dz in [-1,0,1]:
                        if dx==0 and dy==0 and dz==0: 
                            continue
                        dirs.append((dx,dy,dz))

        for (dx,dy,dz) in dirs:
            nx,ny,nz= x+dx,y+dy,z+dz
            if self.in_bounds(nx,ny,nz) and not self.is_blocked(nx,ny,nz):
                yield (nx,ny,nz)

    def _reconstruct_path(self, endpos):
        path=[]
        cur= endpos
        while cur is not None:
            path.append(cur)
            cur= self.came_from[cur]
        path.reverse()
        return path


# ---------------- PotentialField3D ----------------
class PotentialField3D:
    def __init__(self, env, start, goal, max_steps=2000, use_octomap=None):
        self.env= env
        self.start= start
        self.goal= goal
        self.max_steps= max_steps
        self.octomap= use_octomap
        self.steps=0
        self.field= self.compute_field()

        self.jump_when_stuck= False  # 后面可由 config 设置

    def set_jump_when_stuck(self, flag):
        self.jump_when_stuck= flag

    def compute_field(self):
        W,H,D= self.env.width, self.env.height, self.env.depth
        field= np.zeros((D,H,W), dtype=float)
        gx,gy,gz= self.goal
        for z in range(D):
            for y in range(H):
                for x in range(W):
                    dist= math.sqrt((x-gx)**2+(y-gy)**2+(z-gz)**2)
                    field[z,y,x]= -dist
        # 障碍点设置很大势能
        for (ox,oy,oz) in self.env.obstacles:
            if 0<=ox<W and 0<=oy<H and 0<=oz<D:
                field[oz,oy,ox]= 999999
        return field

    def is_blocked(self,x,y,z):
        return self.env.is_blocked(x,y,z)

    def in_bounds(self,x,y,z):
        return self.env.in_bounds(x,y,z)

    def plan(self):
        if self.is_blocked(*self.start) or self.is_blocked(*self.goal):
            return None
        path=[self.start]
        current= self.start
        stuck_count=0
        for _ in range(self.max_steps):
            self.steps+=1
            if current== self.goal:
                return path

            nxt= self.best_neighbor(current)
            if (not nxt) or (nxt== current):
                stuck_count+=1
                if self.jump_when_stuck and stuck_count>10:
                    # 试着随机跳
                    tries=0
                    jumped=False
                    while tries<50:
                        rx= random.randint(0,self.env.width-1)
                        ry= random.randint(0,self.env.height-1)
                        rz= random.randint(0,self.env.depth-1)
                        if not self.is_blocked(rx,ry,rz):
                            nxt= (rx,ry,rz)
                            stuck_count=0
                            jumped=True
                            break
                        tries+=1
                    if not jumped:
                        return None
                else:
                    return None
            else:
                stuck_count=0

            path.append(nxt)
            current= nxt
        return None

    def best_neighbor(self, pos):
        (x,y,z)= pos
        candidates= [(x,y,z)]
        for (dx,dy,dz) in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            nx,ny,nz= x+dx,y+dy,z+dz
            if self.in_bounds(nx,ny,nz) and not self.is_blocked(nx,ny,nz):
                candidates.append((nx,ny,nz))

        def val(px,py,pz):
            return self.field[pz,py,px]
        best= min(candidates, key=lambda c: val(*c))
        return best if best!= pos else None


# ---------------- RRT3D ----------------
class Node3D:
    def __init__(self, x,y,z, parent=None):
        self.x=x
        self.y=y
        self.z=z
        self.parent= parent

def distance_3d(n1,n2):
    return math.sqrt((n1.x-n2.x)**2+(n1.y-n2.y)**2+(n1.z-n2.z)**2)

class RRT3D:
    def __init__(self, env, start, goal,
                 step_size=3, max_iter=1000, goal_threshold=4.0,
                 use_octomap=None, goal_bias=0.0):
        self.env= env
        self.start= Node3D(*start)
        self.goal=  Node3D(*goal)
        self.step_size= step_size
        self.max_iter= max_iter
        self.goal_threshold= goal_threshold
        self.goal_bias= goal_bias
        self.octomap= use_octomap

        self.node_list= [self.start]
        self.steps=0

    def is_blocked(self,x,y,z):
        if not self.env.in_bounds(x,y,z):
            return True
        return self.env.is_blocked(x,y,z)

    def plan(self):
        if self.is_blocked(self.start.x,self.start.y,self.start.z) or \
           self.is_blocked(self.goal.x,self.goal.y,self.goal.z):
            return None

        for _ in range(self.max_iter):
            self.steps+=1

            # goal_bias
            if random.random()< self.goal_bias:
                rx,ry,rz= self.goal.x, self.goal.y, self.goal.z
            else:
                rx= random.randint(0,self.env.width-1)
                ry= random.randint(0,self.env.height-1)
                rz= random.randint(0,self.env.depth-1)

            if self.is_blocked(rx,ry,rz):
                continue
            rnd= Node3D(rx,ry,rz)

            nearest= self.get_nearest_node(rnd)
            new_node= self.steer(nearest, rnd)
            if not self.is_blocked(new_node.x,new_node.y,new_node.z):
                new_node.parent= nearest
                self.node_list.append(new_node)
                if distance_3d(new_node, self.goal)<= self.goal_threshold:
                    return self.build_path(new_node)
        return None

    def get_nearest_node(self, node):
        nearest= None
        min_d= float('inf')
        for nd in self.node_list:
            d= distance_3d(nd,node)
            if d<min_d:
                min_d=d
                nearest= nd
        return nearest

    def steer(self, from_nd, to_nd):
        dx= to_nd.x- from_nd.x
        dy= to_nd.y- from_nd.y
        dz= to_nd.z- from_nd.z
        dist= math.sqrt(dx*dx+dy*dy+dz*dz)
        if dist==0:
            return Node3D(from_nd.x, from_nd.y, from_nd.z, from_nd)
        scale= self.step_size/dist
        nx= int(round(from_nd.x+ dx*scale))
        ny= int(round(from_nd.y+ dy*scale))
        nz= int(round(from_nd.z+ dz*scale))
        return Node3D(nx,ny,nz, from_nd)

    def build_path(self, node):
        path=[]
        cur= node
        while cur is not None:
            path.append((cur.x,cur.y,cur.z))
            cur= cur.parent
        path.reverse()
        return path


# ---------------- RRTStar3D ----------------
class RRTStar3D(RRT3D):
    def __init__(self, env, start, goal,
                 step_size=3, max_iter=1000, goal_threshold=4.0,
                 rewire_radius=3.0, goal_bias=0.0):
        super().__init__(env, start, goal,
                         step_size, max_iter, goal_threshold,
                         goal_bias=goal_bias)
        self.rewire_radius= rewire_radius

    def plan(self):
        if self.is_blocked(self.start.x,self.start.y,self.start.z) or \
           self.is_blocked(self.goal.x,self.goal.y,self.goal.z):
            return None

        cost_map= {self.start: 0.0}
        for _ in range(self.max_iter):
            self.steps+=1

            # goal_bias
            if random.random()< self.goal_bias:
                rx,ry,rz= self.goal.x, self.goal.y, self.goal.z
            else:
                rx= random.randint(0,self.env.width-1)
                ry= random.randint(0,self.env.height-1)
                rz= random.randint(0,self.env.depth-1)

            if self.is_blocked(rx,ry,rz):
                continue
            rnd= Node3D(rx,ry,rz)

            nearest= self.get_nearest_node(rnd)
            new_node= self.steer(nearest, rnd)
            if not self.is_blocked(new_node.x,new_node.y,new_node.z):
                new_node.parent= nearest
                cost_map[new_node]= cost_map[nearest]+ distance_3d(nearest,new_node)
                self.node_list.append(new_node)

                # rewire
                for nd in self.node_list:
                    if nd==new_node or nd==nearest:
                        continue
                    if distance_3d(nd,new_node)<= self.rewire_radius:
                        new_cost= cost_map[new_node]+ distance_3d(new_node, nd)
                        if new_cost< cost_map.get(nd,1e9):
                            nd.parent= new_node
                            cost_map[nd]= new_cost

                if distance_3d(new_node, self.goal)<= self.goal_threshold:
                    return self.build_path(new_node)
        return None


# ---------------- CoarseToFinePlanner ----------------
class CoarseToFinePlanner:
    """
    简易示例:
      1) 在coarse层 用(bfs/astar/rrt)找到粗路径
      2) 对每个粗层节点之间的段, 用fine层(bfs/astar/rrt) refine
    """
    def __init__(self, env, start, goal,
                 neighbor_mode="6",
                 coarse_planner="bfs",
                 fine_planner="astar",
                 octomap=None):
        self.env= env
        self.start= start
        self.goal= goal
        self.neighbor_mode= neighbor_mode
        self.coarse_planner_name= coarse_planner
        self.fine_planner_name= fine_planner
        self.octomap= octomap
        self.steps=0
        self.coarse_path=None

    def plan(self):
        if not self.octomap:
            return None

        # 1) coarse层
        cstart= self.map_to_coarse(self.start)
        cgoal= self.map_to_coarse(self.goal)
        planner_coarse= self.create_planner(self.coarse_planner_name,
                                            cstart, cgoal,
                                            coarse_layer=True, fine_layer=False)
        self.coarse_path= planner_coarse.plan()
        self.steps+= planner_coarse.steps
        if not self.coarse_path:
            return None

        # 2) refine
        full_path=[]
        for i in range(len(self.coarse_path)-1):
            cA= self.coarse_path[i]
            cB= self.coarse_path[i+1]
            Apos= self.coarse_to_center(cA)
            Bpos= self.coarse_to_center(cB)
            planner_fine= self.create_planner(self.fine_planner_name,
                                              Apos, Bpos,
                                              coarse_layer=False, fine_layer=True)
            subpath= planner_fine.plan()
            self.steps+= planner_fine.steps
            if not subpath:
                return None
            if i>0:
                subpath= subpath[1:]
            full_path.extend(subpath)
        return full_path

    def create_planner(self, name, start, goal, coarse_layer=False, fine_layer=False):
        if name=="bfs":
            return BFS3D(self.env, start, goal,
                         neighbor_mode=self.neighbor_mode,
                         use_octomap=self.octomap,
                         coarse_layer=coarse_layer,
                         fine_layer=fine_layer)
        elif name=="astar":
            return AStar3D(self.env, start, goal,
                           neighbor_mode=self.neighbor_mode,
                           heuristic="euclidean",
                           use_octomap=self.octomap,
                           coarse_layer=coarse_layer,
                           fine_layer=fine_layer)
        elif name=="rrt":
            # 这里演示固定step_size等, 你也可再加goal_bias等
            return RRT3D(self.env, start, goal,
                         step_size=3, max_iter=1000, goal_threshold=5.0)
        else:
            raise ValueError(f"Unsupported coarse/fine planner {name}")

    def map_to_coarse(self, pos):
        (x,y,z)= pos
        cx= x// self.octomap.coarse_res
        cy= y// self.octomap.coarse_res
        cz= z// self.octomap.coarse_res
        return (cx,cy,cz)

    def coarse_to_center(self, cpos):
        (cx,cy,cz)= cpos
        base_x= cx*self.octomap.coarse_res
        base_y= cy*self.octomap.coarse_res
        base_z= cz*self.octomap.coarse_res
        half= self.octomap.coarse_res//2
        rx= base_x+ half
        ry= base_y+ half
        rz= base_z+ half
        rx= min(rx,self.env.width-1)
        ry= min(ry,self.env.height-1)
        rz= min(rz,self.env.depth-1)
        return (rx,ry,rz)


class GreedyLinePlanner3D:
    """
    非常简单的“直线趋近”算法：
      1) 在每一步，先计算 (dx, dy, dz) = goal - current
      2) 尝试朝 (sign(dx), sign(dy), sign(dz)) 迈一步
      3) 若该步被障碍阻塞，则在邻域内(可用26邻域或6邻域)寻找一个能让距离变小且没障碍的点
      4) 重复，直到到达目标或达不到
    """

    def __init__(self, env, start, goal, max_steps=2000):
        self.env = env
        self.start = start
        self.goal = goal
        self.max_steps = max_steps
        self.steps = 0

    def plan(self):
        # 如果起点或终点在障碍
        if self.env.is_blocked(*self.start) or self.env.is_blocked(*self.goal):
            return None

        path = [self.start]
        current = self.start
        for _ in range(self.max_steps):
            self.steps += 1
            if current == self.goal:
                return path

            next_pos = self.get_next_position(current)
            if not next_pos or next_pos == current:
                # 找不到更好的前进位置，直接失败
                return None

            path.append(next_pos)
            current = next_pos

        return None  # 超过 max_steps 还没到，就视为失败

    def get_next_position(self, current):
        """
        根据 (dx, dy, dz) 的符号，先尝试往最直接的方向走。
        如果被挡住，则尝试在邻居里找一个更靠近goal的点。
        """
        (cx, cy, cz) = current
        (gx, gy, gz) = self.goal
        dx = gx - cx
        dy = gy - cy
        dz = gz - cz

        # step_x = sign(dx), step_y = sign(dy), step_z = sign(dz)
        step_x = 0 if dx == 0 else (1 if dx>0 else -1)
        step_y = 0 if dy == 0 else (1 if dy>0 else -1)
        step_z = 0 if dz == 0 else (1 if dz>0 else -1)

        primary_choice = (cx + step_x, cy + step_y, cz + step_z)

        # 如果 primary_choice 不被障碍，则直接走
        if not self.env.is_blocked(*primary_choice):
            return primary_choice

        # 否则，我们在邻域(可以用6邻域或26邻域)中找一个更靠近goal的点
        # 这里演示用26邻域
        neighbors = []
        for ddx in [-1, 0, 1]:
            for ddy in [-1, 0, 1]:
                for ddz in [-1, 0, 1]:
                    if ddx==0 and ddy==0 and ddz==0:
                        continue
                    nx, ny, nz = cx+ddx, cy+ddy, cz+ddz
                    if self.env.in_bounds(nx, ny, nz) and not self.env.is_blocked(nx, ny, nz):
                        neighbors.append((nx, ny, nz))

        if not neighbors:
            return None  # 周围全挡住了

        # 在 neighbors 中选一个“让距离到goal变小”的点
        best_nbr = None
        best_dist= self.dist3d(cx,cy,cz, gx,gy,gz)
        for (nx, ny, nz) in neighbors:
            d = self.dist3d(nx, ny, nz, gx, gy, gz)
            if d < best_dist:
                best_dist = d
                best_nbr = (nx, ny, nz)

        return best_nbr if best_nbr else None

    def dist3d(self, x1, y1, z1, x2, y2, z2):
        return ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**0.5
