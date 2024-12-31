import math, random, heapq
import numpy as np
from collections import deque

from octomap import OctoMap

###############################################################################
# 包含 BFS3D, AStar3D, PotentialField3D, RRT3D, RRTStar3D, CoarseToFinePlanner
###############################################################################


# ---------------- BFS3D ----------------
class BFS3D:
    def __init__(self, env, start, goal, neighbor_mode="6",
                 use_octomap=None, coarse_layer=False, fine_layer=False):
        self.env= env
        self.start= start
        self.goal = goal
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

    def compute_field(self):
        W,H,D= self.env.width, self.env.height, self.env.depth
        field= np.zeros((D,H,W), dtype=float)
        gx,gy,gz= self.goal
        for z in range(D):
            for y in range(H):
                for x in range(W):
                    dist= math.sqrt((x-gx)**2+(y-gy)**2+(z-gz)**2)
                    field[z,y,x]= - dist
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
        cur= self.start
        for _ in range(self.max_steps):
            self.steps+=1
            if cur== self.goal:
                return path
            nxt= self.best_neighbor(cur)
            if (not nxt) or (nxt== cur):
                return None
            path.append(nxt)
            cur= nxt
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
                 step_size=3, max_iter=1000, goal_threshold=4.0, use_octomap=None):
        """
        演示版 RRT, 并未接入 octomap
        """
        self.env= env
        self.start= Node3D(*start)
        self.goal=  Node3D(*goal)
        self.step_size= step_size
        self.max_iter= max_iter
        self.goal_threshold= goal_threshold
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
                 rewire_radius=3.0):
        super().__init__(env, start, goal, step_size, max_iter, goal_threshold)
        self.rewire_radius= rewire_radius

    def plan(self):
        if self.is_blocked(self.start.x,self.start.y,self.start.z) or \
           self.is_blocked(self.goal.x,self.goal.y,self.goal.z):
            return None

        cost_map= {self.start: 0.0}
        for _ in range(self.max_iter):
            self.steps+=1
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
    简易演示:
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
