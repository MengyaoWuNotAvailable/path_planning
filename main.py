# main.py

import yaml
import numpy as np
import random
import statistics
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from environment_3d import Environment3D
from octomap import OctoMap
from algorithms_3d import (BFS3D, AStar3D, PotentialField3D, RRT3D, RRTStar3D,
                           CoarseToFinePlanner)

def run_one_experiment(env, algo_name, config, octomap=None):
    """
    返回 (path, time, steps, success_bool)
    """
    # 1) 确定 start / goal
    start = env.random_free_position() if config["start_random"] else tuple(config["start_position"])
    goal  = env.random_free_position() if config["goal_random"]  else tuple(config["goal_position"])

    # 2) 根据 algo_name 创建planner
    if algo_name=="bfs":
        planner = BFS3D(
            env, start, goal,
            neighbor_mode = config["bfs_settings"]["neighbor_mode"],
            use_octomap = octomap,
            coarse_layer=False,
            fine_layer=False
        )
    elif algo_name=="astar":
        planner = AStar3D(
            env, start, goal,
            neighbor_mode = config["astar_settings"]["neighbor_mode"],
            heuristic = config["astar_settings"]["heuristic"],
            use_octomap = octomap,
            coarse_layer=False,
            fine_layer=False
        )
    elif algo_name=="opt":
        planner = PotentialField3D(
            env, start, goal,
            max_steps = config["opt_settings"]["max_steps"]
        )
    elif algo_name=="rrt":
        rrt_cfg = config["rrt_settings"]
        planner = RRT3D(
            env, start, goal,
            step_size = rrt_cfg["step_size"],
            max_iter  = rrt_cfg["max_iter"],
            goal_threshold= rrt_cfg["goal_threshold"]
        )
    elif algo_name=="rrt_star":
        rrt_star_cfg= config["rrt_star_settings"]
        planner = RRTStar3D(
            env, start, goal,
            step_size= rrt_star_cfg["step_size"],
            max_iter=  rrt_star_cfg["max_iter"],
            goal_threshold= rrt_star_cfg["goal_threshold"],
            rewire_radius= rrt_star_cfg["rewire_radius"]
        )
    elif algo_name=="coarse_to_fine":
        c2f_cfg= config["coarse_to_fine_settings"]
        planner = CoarseToFinePlanner(
            env, start, goal,
            neighbor_mode= c2f_cfg["neighbor_mode"],
            refine_method= c2f_cfg["refine_method"],
            octomap= octomap
        )
    else:
        return None, 0.0, 0, False

    # 3) 执行
    t0= time.time()
    path= planner.plan()
    t1= time.time()
    steps= planner.steps
    success= (path is not None)
    return path, (t1 - t0), steps, success

def visualize_paths_3d(env, all_paths, mode="voxel_scatter", title="Paths"):
    """
    mode可选:
     - "voxel_scatter": 障碍物用散点
     - "voxel_plot": Matplotlib voxels函数(适合尺寸较小)
    all_paths: [(algo_name, path, time, steps), ...]
    """
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    if mode=="voxel_scatter":
        # 障碍物散点
        obs = list(env.obstacles)
        ox = [o[0] for o in obs]
        oy = [o[1] for o in obs]
        oz = [o[2] for o in obs]
        ax.scatter(ox, oy, oz, c='black', marker='s', s=20, alpha=0.3, label='Obstacles')
    else:
        # voxel_plot
        grid = np.zeros((env.depth, env.height, env.width), dtype=bool)
        for (ox,oy,oz) in env.obstacles:
            grid[oz, oy, ox] = True
        # Matplotlib的voxel是 [Z,Y,X] 维度
        colors = np.empty(grid.shape, dtype=object)
        colors[grid] = 'black'
        ax.voxels(grid, facecolors=colors, edgecolor='k')

    # 随机分配几种颜色
    import random
    color_list = ["green","blue","red","purple","orange","cyan","lime","yellow"]
    random.shuffle(color_list)

    for i, (algo, path, tcost, steps) in enumerate(all_paths):
        if not path:
            continue
        c = color_list[i % len(color_list)]
        px= [p[0] for p in path]
        py= [p[1] for p in path]
        pz= [p[2] for p in path]
        ax.plot(px, py, pz, color=c, linewidth=2, label=f"{algo} path")
        # 在中点标注
        mid = len(path)//2
        ax.text(px[mid], py[mid], pz[mid], f"{algo}\n{tcost:.2f}s\n{steps}stp", color=c)

    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_zlim(0, env.depth)
    ax.invert_zaxis()  # 可根据喜好
    ax.legend()
    plt.show()

def main():
    with open("config_octomap.yaml","r",encoding="utf-8") as f:
        config = yaml.safe_load(f)

    width  = config["width"]
    height = config["height"]
    depth  = config["depth"]
    obs_count = config["obstacle_count"]
    rnd_seed  = config["random_seed"]

    if rnd_seed is not None:
        random.seed(rnd_seed)
        np.random.seed(rnd_seed)

    # 生成障碍
    obstacles= set()
    for _ in range(obs_count):
        x= random.randint(0,width-1)
        y= random.randint(0,height-1)
        z= random.randint(0,depth-1)
        obstacles.add((x,y,z))

    env= Environment3D(width, height, depth, obstacles)

    # 若 use_octomap
    octomap_obj= None
    if config["use_octomap"]:
        octomap_obj= OctoMap(env,
            coarse_resolution=config["coarse_resolution"],
            fine_resolution=config["fine_resolution"]
        )

    compare= config["compare_algorithms"]
    algo_list= ["bfs","astar","opt","rrt","rrt_star","coarse_to_fine"]
    num_runs= config["num_runs"]
    sampleN= config["sample_paths_for_visualization"]
    vis_mode= config["visualization_mode"]

    if compare:
        # 多算法多次实验
        results= {a: [] for a in algo_list}  # {algo: [(time, steps, success, path), ...], ...}
        for algo in algo_list:
            for run_id in range(num_runs):
                path,tcost,steps,success= run_one_experiment(env, algo, config, octomap_obj)
                results[algo].append((tcost,steps,success, path))

        # 分析
        performance= {}
        for algo in algo_list:
            data= results[algo]
            times= [d[0] for d in data if d[2]]  # 成功样本时间
            steps= [d[1] for d in data if d[2]]
            success_count= sum(1 for d in data if d[2])
            success_rate= success_count/len(data) if len(data)>0 else 0.0

            if times:
                tmean= statistics.mean(times)
                tstd = statistics.pstdev(times) if len(times)>1 else 0
            else:
                tmean= 0
                tstd= 0
            if steps:
                smean= statistics.mean(steps)
                sstd= statistics.pstdev(steps) if len(steps)>1 else 0
            else:
                smean= 0
                sstd= 0

            performance[algo]= (success_rate, tmean,tstd, smean,sstd)

        # 可视化 1: 性能对比
        fig, axs= plt.subplots(3,1, figsize=(8,12))

        X= range(len(algo_list))
        # A) success rate
        sr= [performance[a][0] for a in algo_list]
        axs[0].bar(X, sr, color='green', alpha=0.7)
        axs[0].set_title("Success Rate")
        axs[0].set_ylabel("Rate")
        axs[0].set_xticks(X)
        axs[0].set_xticklabels(algo_list)

        # B) time
        tmeans= [performance[a][1] for a in algo_list]
        tstds=  [performance[a][2] for a in algo_list]
        axs[1].bar(X, tmeans, yerr=tstds, color='blue', alpha=0.7, capsize=5)
        axs[1].set_title("Time (mean ± std on success only)")
        axs[1].set_ylabel("Time (s)")
        axs[1].set_xticks(X)
        axs[1].set_xticklabels(algo_list)

        # C) steps
        smeans= [performance[a][3] for a in algo_list]
        sstds=  [performance[a][4] for a in algo_list]
        axs[2].bar(X, smeans, yerr=sstds, color='orange', alpha=0.7, capsize=5)
        axs[2].set_title("Steps (mean ± std on success only)")
        axs[2].set_ylabel("Steps")
        axs[2].set_xticks(X)
        axs[2].set_xticklabels(algo_list)

        plt.tight_layout()
        plt.show()

        # 可视化 2: 随机抽样路径
        sample_paths= []
        for algo in algo_list:
            data= results[algo]
            success_runs= [d for d in data if d[2] and d[3] is not None]
            random.shuffle(success_runs)
            for i in range(min(sampleN, len(success_runs))):
                tcost,steps,succ,pp= success_runs[i]
                sample_paths.append((algo, pp, tcost, steps))
        if sample_paths:
            visualize_paths_3d(env, sample_paths, mode=vis_mode, title="Sample Paths")
        else:
            print("[INFO] No successful paths to visualize in compare mode.")

    else:
        # 单算法模式
        algo_name= config["algorithm"]
        path,tcost,steps,success= run_one_experiment(env, algo_name, config, octomap_obj)
        if success:
            print(f"{algo_name} Success: time={tcost:.2f}s steps={steps} path_len={len(path)}")
            visualize_paths_3d(env, [(algo_name, path, tcost, steps)], mode=vis_mode, title=algo_name)
        else:
            print(f"{algo_name} Failed to find a path.")
            visualize_paths_3d(env, [], mode=vis_mode, title=f"{algo_name} (No Path)")

if __name__=="__main__":
    main()
