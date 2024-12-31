import yaml
import random
import numpy as np
import statistics
import time
import matplotlib.pyplot as plt

from environment_3d import Environment3D
from octomap import OctoMap
from algorithms_3d import (BFS3D, AStar3D, PotentialField3D,
                           RRT3D, RRTStar3D, CoarseToFinePlanner)

def run_one_experiment(env, algo_name, config, octomap=None):
    """
    在给定 env、algo_name 下执行一次规划。
    返回: (path, time_cost, steps, success, start_pt, goal_pt)
    """
    # 1) start/goal
    if config["start_random"]:
        start= env.random_free_position()
    else:
        start= tuple(config["start_position"])

    if config["goal_random"]:
        goal= env.random_free_position()
    else:
        goal= tuple(config["goal_position"])

    # 2) 创建算法
    if algo_name=="bfs":
        planner= BFS3D(env, start, goal,
                       neighbor_mode=config["bfs_settings"]["neighbor_mode"],
                       use_octomap=octomap, coarse_layer=False, fine_layer=False)

    elif algo_name=="astar":
        planner= AStar3D(env, start, goal,
                         neighbor_mode=config["astar_settings"]["neighbor_mode"],
                         heuristic=config["astar_settings"]["heuristic"],
                         use_octomap=octomap, coarse_layer=False, fine_layer=False)

    elif algo_name=="opt":
        # 势场法
        planner= PotentialField3D(env, start, goal,
                                  max_steps=config["opt_settings"]["max_steps"])
        # 若有jump_when_stuck
        if "jump_when_stuck" in config["opt_settings"]:
            planner.set_jump_when_stuck(config["opt_settings"]["jump_when_stuck"])

    elif algo_name=="rrt":
        rrt_cfg= config["rrt_settings"]
        planner= RRT3D(env, start, goal,
                       step_size= rrt_cfg["step_size"],
                       max_iter= rrt_cfg["max_iter"],
                       goal_threshold= rrt_cfg["goal_threshold"],
                       goal_bias= rrt_cfg.get("goal_bias", 0.0)
                      )

    elif algo_name=="rrt_star":
        rrt_star_cfg= config["rrt_star_settings"]
        planner= RRTStar3D(env, start, goal,
                           step_size= rrt_star_cfg["step_size"],
                           max_iter=  rrt_star_cfg["max_iter"],
                           goal_threshold= rrt_star_cfg["goal_threshold"],
                           rewire_radius= rrt_star_cfg["rewire_radius"],
                           goal_bias= rrt_star_cfg.get("goal_bias", 0.0)
                          )

    elif algo_name=="coarse_to_fine":
        c2f= config["coarse_to_fine_settings"]
        planner= CoarseToFinePlanner(env, start, goal,
                                     neighbor_mode=c2f["neighbor_mode"],
                                     coarse_planner=c2f["coarse_planner"],
                                     fine_planner= c2f["fine_planner"],
                                     octomap= octomap)
    else:
        return None,0.0,0,False,start,goal

    # 3) 规划
    t0= time.time()
    path= planner.plan()
    t1= time.time()
    steps= planner.steps
    success= (path is not None)
    return path, (t1 - t0), steps, success, start, goal


def visualize_paths_3d(env, all_paths, mode="voxel_scatter",
                       title="Paths in one environment", show_start_goal=True):
    """
    在同一个 3D 图中展示多条路径 (各算法对比).
    all_paths: [ (algo_name, path, time_cost, steps, start_pt, goal_pt), ... ]
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig= plt.figure(figsize=(10,8))
    ax= fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    # 绘制障碍
    if mode=="voxel_scatter":
        obs= list(env.obstacles)
        ox= [o[0] for o in obs]
        oy= [o[1] for o in obs]
        oz= [o[2] for o in obs]
        ax.scatter(ox, oy, oz, c='black', marker='s', s=15, alpha=0.3, label='Obstacles')
    else:
        # voxel_plot
        grid= np.zeros((env.depth, env.height, env.width), dtype=bool)
        for (ox,oy,oz) in env.obstacles:
            grid[oz,oy,ox]= True
        colors= np.empty(grid.shape, dtype=object)
        colors[grid]= 'black'
        ax.voxels(grid, facecolors=colors, edgecolor='k')

    color_list= ["red","blue","green","orange","purple","cyan","lime","yellow"]
    for i,(algo, path, tcost, steps, st_pt, gl_pt) in enumerate(all_paths):
        c= color_list[i % len(color_list)]
        if path:
            px= [p[0] for p in path]
            py= [p[1] for p in path]
            pz= [p[2] for p in path]
            ax.plot(px, py, pz, color=c, linewidth=2, label=algo)

            # 在路径中点加标注
            mid= len(path)//2
            ax.text(px[mid], py[mid], pz[mid],
                    f"{algo}\n{tcost:.2f}s\n{steps}stp", color=c)

        # 标出起点(绿三角) & 终点(红叉)
        if show_start_goal:
            ax.scatter([st_pt[0]], [st_pt[1]], [st_pt[2]],
                       marker='^', c='green', s=60)
            ax.scatter([gl_pt[0]], [gl_pt[1]], [gl_pt[2]],
                       marker='x', c='red',   s=60)

    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_zlim(0, env.depth)
    ax.invert_zaxis()
    ax.legend()
    plt.show()


def main():
    # 读取 config
    with open("config_octomap.yaml","r",encoding="utf-8") as f:
        config= yaml.safe_load(f)

    # 如果需要固定随机种子
    rnd_seed= config["random_seed"]
    if rnd_seed is not None:
        random.seed(rnd_seed)
        np.random.seed(rnd_seed)

    # 算法列表
    if config["compare_algorithms"]:
        algo_list= ["bfs","astar","opt","rrt","rrt_star","coarse_to_fine"]
    else:
        algo_list= [config["algorithm"]]

    num_runs= config["num_runs"]

    # 整体性能记录: perf[algo] = [(success, time, steps), ...]
    perf= {a:[] for a in algo_list}

    for env_id in range(num_runs):
        # 1) 生成随机环境
        w= config["width"]
        h= config["height"]
        d= config["depth"]
        obs_count= config["obstacle_count"]
        obstacles= set()
        for _ in range(obs_count):
            rx= random.randint(0,w-1)
            ry= random.randint(0,h-1)
            rz= random.randint(0,d-1)
            obstacles.add((rx,ry,rz))
        env= Environment3D(w,h,d, obstacles)

        # 2) 若 use_octomap
        octomap_obj= None
        if config["use_octomap"]:
            octomap_obj= OctoMap(env,
                                 coarse_resolution=config["coarse_resolution"],
                                 fine_resolution=config["fine_resolution"])

        # 3) 在该环境中运行每个算法
        run_results=[]
        for algo in algo_list:
            path,tc,steps,success,st_pt,gl_pt= run_one_experiment(env, algo, config, octomap_obj)
            run_results.append((algo, path, tc, steps, st_pt, gl_pt))

            # 记录性能
            perf[algo].append((success, tc, steps))

        # 4) 可视化本环境下所有算法的轨迹 (同一张图)
        visualize_paths_3d(env, run_results,
                           mode=config["visualization_mode"],
                           title=f"Environment #{env_id+1}",
                           show_start_goal=config["show_start_goal_in_vis"])

    # 所有环境跑完，做统计
    print("\n=== Final Statistics over all environments ===")
    algo_list_sorted= list(perf.keys())
    for algo in algo_list_sorted:
        data= perf[algo]
        success_count= sum(1 for d in data if d[0])
        sr= success_count/ len(data) if data else 0
        times= [d[1] for d in data if d[0]]
        steps= [d[2] for d in data if d[0]]
        if times:
            tmean= statistics.mean(times)
            tstd= statistics.pstdev(times) if len(times)>1 else 0
        else:
            tmean= 0; tstd= 0
        if steps:
            smean= statistics.mean(steps)
            sstd= statistics.pstdev(steps) if len(steps)>1 else 0
        else:
            smean= 0; sstd= 0

        print(f" - {algo}: successRate={sr*100:.1f}% "
              f"time={tmean:.3f}±{tstd:.3f}, steps={smean:.1f}±{sstd:.1f}")

    # 统计结果可视化 (柱状图)
    fig, axs= plt.subplots(3,1, figsize=(8,12))
    X= range(len(algo_list_sorted))

    # A) 成功率
    sr_vals= []
    for algo in algo_list_sorted:
        data= perf[algo]
        sc= sum(1 for d in data if d[0])
        sr= sc/len(data) if len(data)>0 else 0
        sr_vals.append(sr)
    axs[0].bar(X, sr_vals, color='green', alpha=0.7)
    axs[0].set_title("Success Rate")
    axs[0].set_ylabel("Rate")
    axs[0].set_xticks(list(X))
    axs[0].set_xticklabels(algo_list_sorted)

    # B) 平均时间
    tmeans=[]
    tstds=[]
    for algo in algo_list_sorted:
        data= perf[algo]
        t= [d[1] for d in data if d[0]]
        if t:
            m= statistics.mean(t)
            s= statistics.pstdev(t) if len(t)>1 else 0
        else:
            m=0; s=0
        tmeans.append(m)
        tstds.append(s)
    axs[1].bar(X, tmeans, yerr=tstds, color='blue', alpha=0.7, capsize=5)
    axs[1].set_title("Time (mean ± std, success only)")
    axs[1].set_ylabel("Seconds")
    axs[1].set_xticks(list(X))
    axs[1].set_xticklabels(algo_list_sorted)

    # C) 平均步数
    smeans=[]
    sstds=[]
    for algo in algo_list_sorted:
        data= perf[algo]
        st= [d[2] for d in data if d[0]]
        if st:
            m= statistics.mean(st)
            s= statistics.pstdev(st) if len(st)>1 else 0
        else:
            m=0; s=0
        smeans.append(m)
        sstds.append(s)
    axs[2].bar(X, smeans, yerr=sstds, color='orange', alpha=0.7, capsize=5)
    axs[2].set_title("Steps (mean ± std, success only)")
    axs[2].set_ylabel("Steps")
    axs[2].set_xticks(list(X))
    axs[2].set_xticklabels(algo_list_sorted)

    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()
