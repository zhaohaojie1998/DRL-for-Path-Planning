# -*- coding: utf-8 -*-
"""
路径规划强化学习环境
 Created on Tue May 16 2023 17:54:17
 Modified on 2023-8-02 17:38:00
 
 @auther: HJ https://github.com/zhaohaojie1998
"""
#
import gym
import math
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
from copy import deepcopy
from pathlib import Path
from collections import deque
from scipy.integrate import odeint
from shapely import geometry as geo
from shapely.plotting import plot_polygon

__all__ = ["DynamicPathPlanning", "StaticPathPlanning", "NormalizedActionsWrapper"]



#----------------------------- ↓↓↓↓↓ 参数设置 ↓↓↓↓↓ ------------------------------#
# 地图设置
class MAP:
    size = [[-10.0, -10.0], [10.0, 10.0]] # x, z最小值; x, z最大值
    start_pos = [0, -9]                   # 起点坐标
    end_pos = [2.5, 9]                    # 终点坐标
    obstacles = [                         # 障碍物, 要求为 geo.Polygon 或 带buffer的 geo.Point/geo.LineString
        geo.Point(0, 2.5).buffer(4),
        geo.Point(-6, -5).buffer(3),
        geo.Point(6, -5).buffer(3),
        geo.Polygon([(-10, 0), (-10, 5), (-7.5, 5), (-7.5, 0)])
    ]

    @classmethod
    def show(cls):
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.close('all')
        fig, ax = plt.subplots()
        ax.clear()
        ax.set_aspect('equal')
        ax.set_xlim(cls.size[0][0], cls.size[1][0])
        ax.set_ylim(cls.size[0][1], cls.size[1][1])
        ax.invert_yaxis()
        for o in cls.obstacles:
            plot_polygon(o, facecolor='w', edgecolor='k', add_points=False)
        ax.scatter(cls.start_pos[0], cls.start_pos[1], s=30, c='k', marker='x', label='起点')
        ax.scatter(cls.end_pos[0], cls.end_pos[1], s=30, c='k', marker='o', label='终点')
        ax.legend(loc='best').set_draggable(True)
        ax.set_title('Map')
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.grid(alpha=0.3, ls=':')
        plt.show(block=True)

        
# 运动速度设置
V_LOW = 0.05
V_HIGH = 0.2
# 质心动力学状态设置
STATE_LOW = [MAP.size[0][0], MAP.size[0][1], V_LOW, -math.pi] # x, z, V, ψ
STATE_HIGH = [MAP.size[1][0], MAP.size[1][1], V_HIGH, math.pi] # x, z, V, ψ
# 观测设置
OBS_STATE_LOW = [0, -math.pi, V_LOW]                                                                  # 相对终点距离 + 相对终点方位角 rad + 速度
OBS_STATE_HIGH = [1.414*max(STATE_HIGH[0]-STATE_LOW[0], STATE_HIGH[1]-STATE_LOW[1]), math.pi, V_HIGH] # 相对终点距离 + 相对终点方位角 rad + 速度
# 控制设置
CTRL_LOW = [-0.02, -math.pi/3] # 切向过载 + 速度滚转角 rad
CTRL_HIGH = [0.02, math.pi/3]  # 切向过载 + 速度滚转角 rad
# 雷达设置
SCAN_RANGE = 5   # 扫描距离
SCAN_ANGLE = 128 # 扫描范围 deg
SCAN_NUM = 128   # 扫描点个数
# 碰撞半径
SAFE_DISTANCE = 0.5
# 时间序列长度
TIME_STEP = 4





#----------------------------- ↓↓↓↓↓ 避障控制环境 ↓↓↓↓↓ ------------------------------#
if __name__ == '__main__':
    from lidar_sim import LidarModel
else:
    from .lidar_sim import LidarModel

class Logger:
    pass

class DynamicPathPlanning(gym.Env):
    """从力学与控制的角度进行规划
    >>> dx/dt = V * cos(ψ)
    >>> dz/dt = -V * sin(ψ)
    >>> dV/dt = g * nx
    >>> dψ/dt = -g / V * tan(μ)
    >>> u = [nx, μ]
    obs_state space = [d_start, ε_start, d_end, ε_end, V, ψ]
    """

    def __init__(self, max_episode_steps=200, dt=0.5, use_old_gym=True):
        """最大回合步数, 决策周期, 是否采用老版接口
        """
        # 仿真
        self.dt = dt
        self.time_step = 0
        self.max_episode_steps = max_episode_steps
        self.logger = Logger
        # 地图
        self.map = MAP
        self.obstacles = MAP.obstacles
        self.lidar = LidarModel(SCAN_RANGE, SCAN_ANGLE, SCAN_NUM)
        self.lidar.add_obstacles(MAP.obstacles)
        self.safe_distance = SAFE_DISTANCE
        # 状态空间 + 控制空间
        self.state_space = spaces.Box(np.array(STATE_LOW), np.array(STATE_HIGH))
        self.control_space = spaces.Box(np.array(CTRL_LOW), np.array(CTRL_HIGH))
        # 观测空间 + 动作空间
        obs_points = spaces.Box(0, SCAN_RANGE, (TIME_STEP, SCAN_NUM, )) # seq_len, dim
        obs_vector = spaces.Box(np.array([OBS_STATE_LOW]*TIME_STEP), np.array([OBS_STATE_HIGH]*TIME_STEP)) # seq_len, dim
        self.observation_space = spaces.Dict({'seq_points': obs_points, 'seq_vector': obs_vector})
        self.action_space = spaces.Box(-1, 1, (len(CTRL_LOW), ))
        # 序列观测
        self.deque_points = deque(maxlen=TIME_STEP)
        self.deque_vector = deque(maxlen=TIME_STEP)
        # 环境控制
        self.__need_reset = True
        self.__old_gym = use_old_gym
    
    def _get_u(self, a, u_last=None, tau=None):
        lb = self.control_space.low
        ub = self.control_space.high
        u = lb + (a + 1.0) * 0.5 * (ub - lb) # [-1,1] -> [lb,ub]
        u = np.clip(u, lb, ub)
        # smooth control signal u
        if u_last is not None and tau is not None:
            u = (1. - tau) * u_last + tau * u
        return u

    def reset(self, mode=0):
        """mode=0, 随机初始化起点终点
           mode=1, 初始化起点终点到地图设置
        """
        self.__need_reset = False
        obs, info = None, {}
        if self.__old_gym:
            return obs
        return obs, info
    
    def step(self, act):
        assert not self.__need_reset, "调用step前必须先reset"

        obs, rew, done, truncated, info = None, None, None, None, {}

        if truncated or done:
            info["terminal"] = True
            self.__need_reset = True
        else:
            info["terminal"] = False

        if self.__old_gym:
            return obs, rew, done, info
        return obs, rew, done, truncated, info
    
    def plot(self, file):
        """绘图输出"""

    
    def render(self, mode="human"):
        """可视化环境, 和step交替调用"""
        assert not self.__need_reset, "调用render前必须先reset"
        frame = None
        self.__update_frame(frame)

    def __update_frame(self, frame):
        """更新动画"""
        pass

    def close(self): 
        """关闭环境"""
        self.__need_reset = True
        plt.close()

    @staticmethod
    def _normalize_points(points, max_range=SCAN_RANGE):
        """归一化雷达测距 (无效数据变成0, 有效数据0.1~1)"""
        points = np.array(points)
        points[points>-0.5] = 0.9*points[points>-0.5]/max_range + 0.1
        points[points<-0.5] = 0.0
        return points
    
    @staticmethod
    def _limit_angle(x, domain=1):
        """限制角度 x 的区间: 1限制在(-π, π], 2限制在[0, 2π)"""
        x = x - x//(2*math.pi) * 2*math.pi # any -> [0, 2π)
        if domain == 1 and x > math.pi:
            return x - 2*math.pi           # [0, 2π) -> (-π, π]
        return x

    @staticmethod
    def _linear_mapping(x, x_min, x_max, left=0.0, right=1.0):  
        """x 线性变换: [x_min, x_max] -> [left, right]"""
        y = left + (right - left) / (x_max - x_min) * (x - x_min)
        return y
    
    @staticmethod
    def _vector_angle(x_vec, y_vec, EPS=1e-8):
        """计算向量 x_vec 与 y_vec 之间的夹角 [0, π]"""
        x = np.linalg.norm(x_vec) * np.linalg.norm(y_vec)
        y = np.dot(x_vec, y_vec)
        if x < EPS: # 0向量情况
            return 0.0
        if y < EPS: # 90°情况
            return math.pi/2
        return math.acos(np.clip(y/x, -1, 1)) # note: x很小的时候, 可能会超过+-1
    
    @staticmethod
    def _compute_azimuth(pos1, pos2, use_3d_pos=False):
        """计算pos2相对pos1的方位角 [-π, π] 和高度角(3D情况) [-π/2, π/2] """
        if use_3d_pos:
            x, y, z = np.array(pos2) - pos1
            q = math.atan(y / (math.sqrt(x**2 + z**2) + 1e-8)) # 高度角 [-π/2, π/2]
            ε = math.atan2(-z, x)                              # 方位角 [-π, π]
            return ε, q
        else:
            x, z = np.array(pos2) - pos1
            return math.atan2(-z, x)
    
    @staticmethod
    def _fixed_wing_2d(s, t, u):
        """平面运动ode模型
        s = [x, z, V, ψ]
        u = [nx, μ]
        """
        _, _, V, ψ = s
        nx, μ = u
        dsdt = [
            V * math.cos(ψ),
            -V * math.sin(ψ),
            9.8 * nx,
            -9.8/V * math.tan(μ) # μ<90, 不存在inf情况
        ]
        return dsdt
    
    @staticmethod
    def _fixed_wing_3d(s, t, u):
        """空间运动ode模型
        s = [x, y, z, V, θ, ψ]
        u = [nx, ny, μ]
        """
        _, _, _, V, θ, ψ = s
        nx, ny, μ = u
        if abs(math.cos(θ)) < 0.01:
            dψdt = 0 # θ = 90° 没法积分了!!!
        else:
            dψdt = -9.8 * ny*math.sin(μ) / (V*math.cos(θ))
        dsdt = [
            V * math.cos(θ) * math.cos(ψ),
            V * math.sin(θ),
            -V * math.cos(θ) * math.sin(ψ),
            9.8 * (nx - math.sin(θ)),
            9.8/V * (ny*math.cos(μ) - math.cos(θ)),
            dψdt
        ]
        return dsdt
    
    @classmethod
    def _ode45(cls, s_old, u, dt, use_3d_model=False):
        """微分方程积分"""
        model = cls._fixed_wing_3d if use_3d_model else cls._fixed_wing_2d
        s_new = odeint(model, s_old, (0.0, dt), args=(u, )) # shape=(len(t), len(s))
        return np.array(s_new[-1]) # deepcopy

  
    


#----------------------------- ↓↓↓↓↓ 路径搜索环境 ↓↓↓↓↓ ------------------------------#
class StaticPathPlanning(gym.Env):
    """从航点搜索的角度进行规划"""

    def __init__(self, num_pos=6, max_search_steps=200, use_old_gym=True):
        """起点终点之间的航点个数, 最大搜索次数, 是否采用老版接口
        """
        self.num_pos = num_pos
        self.map = MAP
        self.max_episode_steps = max_search_steps

        lb = np.array(self.map.size[0] * num_pos)
        ub = np.array(self.map.size[1] * num_pos)
        self.observation_space = spaces.Box(lb, ub, dtype=np.float32)
        self.action_space = spaces.Box(lb/10, ub/10, dtype=np.float32)

        self.__render_not_called = True
        self.__need_reset = True
        self.__old_gym = use_old_gym

    def reset(self):
        self.__need_reset = False
        self.time_steps = 0 # NOTE: 容易简写成step, 会和step method重名, 还不容易发现 BUG
        self.obs = self.observation_space.sample()
        # New Gym: obs, info
        # Old Gym: obs
        if self.__old_gym:
            return self.obs
        return self.obs, {}
    
    def step(self, act):
        """
        转移模型 1
        Pos_new = act = Actor(Pos_old)
        Q = Critic(Pos_old, act) = Critic(Pos_old, Pos_new)
        转移模型 2
        Pos_new = Pos_old + act, act = Actor(Pos_old)
        Q = Critic(Pos_old, act) = Critic(Pos_old, Pos_new-Pos_old)
        """
        assert not self.__need_reset, "调用step前必须先reset"
        # 状态转移
        obs = np.clip(self.obs + act, self.observation_space.low, self.observation_space.high)
        self.time_steps += 1
        # 计算奖励
        rew, done, info = self.get_reward(obs)
        # 回合终止
        truncated = self.time_steps >= self.max_episode_steps
        if truncated or done:
            info["terminal"] = True
            self.__need_reset = True
        else:
            info["terminal"] = False
        # 更新状态
        self.obs = deepcopy(obs)
        # New Gym: obs, rew, done, truncated, info
        # Old Gym: obs, rew, done, info
        if self.__old_gym:
            return obs, rew, done, info
        return obs, rew, done, truncated, info
    
    def get_reward(self, obs):
        traj = np.array(self.map.start_pos + obs.tolist() + self.map.end_pos) # [x,y,x,y,x,y,...]
        traj = traj.reshape(self.num_pos+2, 2) # [[x,y],[x,y],...]
        # 惩罚
        num_over = 0 # 超出约束的状态个数
        for o, l, u in zip(obs, self.observation_space.low, self.observation_space.high):
            if o <= l or o >= u:
                num_over += 2
        # 轨迹检测
        d = 0.0      # 总长度 -> 衡量能量
        dθall = 0.0  # 总的角度变化 -> 衡量能量
        θ_last = 0.0 # 转角变化 < 45度
        num_theta = 0 # 不合理转角次数 -> 惩罚
        num_crash = 0 # 碰撞次数 -> 惩罚
        for i in range(len(traj) - 1):
            vec = traj[i+1] - traj[i]   # 轨迹向量 (2, )
            # 累积长度
            d += np.linalg.norm(vec) 
            # 转角检测
            θ = math.atan2(vec[1], vec[0])
            dθ = abs(θ - θ_last) if i !=0 else 0.0 # 第一段没转角变化
            if dθ >= math.pi/4: 
                num_theta += 1
            dθall += dθ
            θ_last = deepcopy(θ)
            # 碰撞检测
            for o in self.map.obstacles:
                line = geo.LineString(traj[i:i+2])
                if o.intersects(line): # 判断是否有交集
                    num_crash += 1
            #end 
        #end
        # 总奖励
        rew = -d -dθall/self.num_pos -num_theta -num_crash -num_over
        # 是否终止
        if num_theta == 0 and num_crash == 0:
            rew += 100  # 给个终端奖励
            done = True # 轨迹合理
        else:
            done = False # 轨迹不合理
        info = {}
        info['碰撞次数'] = num_crash
        info['不平滑次数'] = num_theta
        info['done'] = done
        return rew, done, info
    
    def render(self, mode="human"):
        """环境可视化, 和step交替调用"""
        assert not self.__need_reset, "调用render前必须先reset"
        if self.__render_not_called:
            self.__render_not_called = False
            plt.ion() # 打开交互绘图, 只能开一次
        # 创建窗口
        plt.clf()
        plt.axis('equal')
        plt.xlim(self.map.size[0][0], self.map.size[1][0])
        plt.ylim(self.map.size[1][1], self.map.size[0][1]) # NOTE min/max调换顺序可反转坐标轴
        # 绘制
        for o in self.map.obstacles:
            plot_polygon(o, facecolor='w', edgecolor='k', add_points=False)
        plt.scatter(self.map.start_pos[0], self.map.start_pos[1], s=30, c='k', marker='x', label='起点')
        plt.scatter(self.map.end_pos[0], self.map.end_pos[1], s=30, c='k', marker='o', label='终点')
        traj = self.map.start_pos + self.obs.tolist() + self.map.end_pos # [x,y,x,y,x,y,]
        plt.plot(traj[::2], traj[1::2], label='path', color='b')
        # 设置信息
        plt.title('Path Planning')
        plt.legend(loc='best')
        plt.xlabel("x")
        plt.ylabel("z")
        plt.grid(alpha=0.3, ls=':')
        # 关闭窗口
        plt.pause(0.001)
        plt.ioff()

    def close(self):
        """关闭环境"""
        self.__render_not_called = True
        self.__need_reset = True
        plt.close()








#----------------------------- ↓↓↓↓↓ 环境-算法适配 ↓↓↓↓↓ ------------------------------#
class NormalizedActionsWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(NormalizedActionsWrapper, self).__init__(env)
        assert isinstance(env.action_space, spaces.Box), '只用于Box动作空间'
  
    # 将神经网络输出转换成gym输入形式
    def action(self, action): 
        # 连续情况 scale action [-1, 1] -> [lb, ub]
        lb, ub = self.action_space.low, self.action_space.high
        action = lb + (action + 1.0) * 0.5 * (ub - lb)
        action = np.clip(action, lb, ub)
        return action

    # 将gym输入形式转换成神经网络输出形式
    def reverse_action(self, action):
        # 连续情况 normalized action [lb, ub] -> [-1, 1]
        lb, ub = self.action_space.low, self.action_space.high
        action = 2 * (action - lb) / (ub - lb) - 1
        return np.clip(action, -1.0, 1.0)
       




if __name__ == '__main__':
    MAP.show()