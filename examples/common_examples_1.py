"""
@file: common_examples.py
@breif: Examples of Python Motion Planning library
@author: Wu Maojia
@update: 2025.4.11
"""
import sys, os

from tools.gen_block import gen_block

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python_motion_planning import *
import matplotlib


if __name__ == '__main__':


    # Create environment with custom obstacles
    # grid_env = Grid(102, 102)
    # grid_env = Grid(152, 152)
    # grid_env = Grid(202, 202)
    grid_env = Grid(302, 302)
    obstacles = grid_env.obstacles
    # for i in range(3, 21):
    #     obstacles.add((i, 15))
    # for i in range(15):
    #     obstacles.add((20, i))
    # for i in range(15, 30):
    #     obstacles.add((30, i))
    # for i in range(16):
    #     obstacles.add((40, i))

    # for i in range(3, 21):
    #     obstacles.add((i, 15))
    # for i in range(15):
    #     obstacles.add((20, i))
    # for i in range(10, 30):
    #     obstacles.add((25, i))
    # for i in range(21):
    #     obstacles.add((40, i))
    # for i in range(100-21):
    #     obstacles.add((70, i))

    starts = [10,30,50,70]
    ends = [10,30,50,70]
    # ends = [20,40,60,80]

    for start in starts:
        for end in ends:
            blocks = gen_block(start=(start,end),end= (start+10,end+10))
            [obstacles.add(i) for i in blocks]
        #     break
        # break

    grid_env.update(obstacles)

    map_env = Map(51, 31)
    obs_rect = [
        [14, 12, 8, 2],
        [18, 22, 8, 3],
        [26, 7, 2, 12],
        [32, 14, 10, 2]
    ]
    obs_circ = [
        [7, 12, 3],
        [46, 20, 2],
        [15, 5, 2],
        [37, 7, 3],
        [37, 23, 3]
    ]
    map_env.update(obs_rect=obs_rect, obs_circ=obs_circ)

    start = (5, 5)
    goal = (85, 95)
    goal = (285, 295)
    # matplotlib.use('Agg')
    # -------------global planners-------------
    # plt = Dijkstra(start=start, goal=goal, env=grid_env)
    # plt = GBFS(start=start, goal=goal, env=grid_env)
    # plt = AStar(start=start, goal=goal, env=grid_env)
    # plt = ThetaStar(start=start, goal=goal, env=grid_env)
    # plt = DStar(start=start, goal=goal, env=grid_env)
    # plt = DStarLite(start=start, goal=goal, env=grid_env)
    # plt = JPS(start=start, goal=goal, env=grid_env)
    # plt = LazyThetaStar(start=start, goal=goal, env=grid_env)
    plt = SThetaStar(start=start, goal=goal, env=grid_env)
    # plt = LPAStar(start=start, goal=goal, env=grid_env)
    # plt = VoronoiPlanner(start=start, goal=goal, env=grid_env)

    # plt = RRT(start=(18, 8), goal=(37, 18), env=map_env)
    # plt = RRTConnect(start=(18, 8), goal=(37, 18), env=map_env)
    # plt = RRTStar(start=(18, 8), goal=(37, 18), env=map_env)
    # plt = InformedRRT(start=(18, 8), goal=(37, 18), env=map_env)

    # plt = ACO(start=start, goal=goal, env=grid_env)
    # plt = PSO(start=start, goal=goal, env=grid_env,max_iter=100)
    plt.run()

    # -------------local planners-------------
    # plt = PID(start=(5, 5, 0), goal=(45, 25, 0), env=grid_env)
    # plt = DWA(start=(5, 5, 0), goal=(45, 25, 0), env=grid_env)
    # plt = APF(start=(5, 5, 0), goal=(45, 25, 0), env=grid_env)
    # plt = LQR(start=(5, 5, 0), goal=(45, 25, 0), env=grid_env)
    # plt = RPP(start=(5, 5, 0), goal=(45, 25, 0), env=grid_env)
    # plt = MPC(start=(5, 5, 0), goal=(45, 25, 0), env=grid_env)
    # plt.run()

    # -------------curve generators-------------
    # points = [(0, 0, 0), (10, 10, -90), (20, 5, 60), (30, 10, 120),
    #           (35, -5, 30), (25, -10, -120), (15, -15, 100), (0, -10, -90)]
    #
    # plt = Dubins(step=0.1, max_curv=0.25)
    # plt = Dubins(step=0.1, max_curv=0.5)
    # plt = Dubins(step=0.1, max_curv=1)
    # plt = Dubins(step=0.1, max_curv=0.1)
    # plt = Bezier(step=0.1, offset=3.0)
    # plt = Polynomial(step=2, max_acc=1.0, max_jerk=0.5)
    # plt = ReedsShepp(step=0.1, max_curv=0.25)
    # plt = CubicSpline(step=0.1)
    # plt = BSpline(step=0.01, k=3)

    # plt.run(points)