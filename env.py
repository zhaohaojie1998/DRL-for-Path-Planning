# -*- coding: utf-8 -*-
"""
 Created on Tue May 16 2023 17:54:17
 Modified on 2023-8-02 17:38:00
 
 @auther: HJ https://github.com/zhaohaojie1998
"""
#

# 路径规划强化学习环境
import gym
import math
import pylab as pl
from gym import spaces
from copy import deepcopy
from collections import deque
from scipy.integrate import odeint
from shapely import geometry as geo
from shapely.plotting import plot_polygon

__all__ = ["StaticPathPlanning", "DynamicPathPlanning", "NormalizedActionsWrapper"]


class Map:
    size = [[-10.0, -10.0], [10.0, 10.0]] # x, y最小值; x, y最大值
    start_pos = [0, -9]           # 起点坐标
    end_pos = [2.5, 9]            # 终点坐标
    obstacles = [                 # 障碍物, 要求为 geo.Polygon 或 带buffer的 geo.Point/geo.LineString
        geo.Point(0, 2.5).buffer(4),
        geo.Point(-6, -5).buffer(3),
        geo.Point(6, -5).buffer(3),
        geo.Polygon([(-10, 0), (-10, 5), (-7.5, 5), (-7.5, 0)])
    ]

    @classmethod
    def show(cls):
        pl.mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 修复字体bug
        pl.mpl.rcParams['axes.unicode_minus'] = False            # 用来正常显示负号

        pl.close('all')
        pl.figure('Map')
        pl.clf()
    
        # 障碍物
        for o in cls.obstacles:
            plot_polygon(o, facecolor='w', edgecolor='k', add_points=False)

        # 起点终点
        pl.scatter(cls.start_pos[0], cls.start_pos[1], s=30, c='k', marker='x', label='起点')
        pl.scatter(cls.end_pos[0], cls.end_pos[1], s=30, c='k', marker='o', label='终点')
  
        pl.legend(loc='best').set_draggable(True) # 显示图例
        pl.axis('equal')                          # 平均坐标
        pl.xlabel("x")                            # x轴标签
        pl.ylabel("y")                            # y轴标签
        pl.xlim(cls.size[0][0], cls.size[1][0])   # x范围
        pl.ylim(cls.size[0][1], cls.size[1][1])   # y范围
        pl.title('Map')                           # 标题
        pl.grid()                                 # 生成网格
        pl.grid(alpha=0.3,ls=':')                 # 改变网格透明度，和网格线的样式、、
        pl.show(block=True)

        


# 静态环境
class StaticPathPlanning(gym.Env):
    """从航点搜索的角度进行规划"""

    # 地图设置
    MAP = Map()

    def __init__(self, num_pos=6, max_search_steps=200, use_old_gym=True):
        """起点终点之间的导航点个数, 最大搜索次数, 是否采用老版接口
        """
        self.num_pos = num_pos
        self.map = self.MAP
        self.max_episode_steps = max_search_steps

        lb = pl.array(self.map.size[0] * num_pos)
        ub = pl.array(self.map.size[1] * num_pos)
        self.observation_space = spaces.Box(lb, ub, dtype=pl.float32)
        self.action_space = spaces.Box(lb/10, ub/10, dtype=pl.float32)

        self.__render_flag = True
        self.__reset_flag = True
        self.__old_gym = use_old_gym

    def reset(self):
        self.__reset_flag = False
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
        assert not self.__reset_flag, "调用step前必须先reset"
        # 状态转移
        obs = pl.clip(self.obs + act, self.observation_space.low, self.observation_space.high)
        self.time_steps += 1
        # 计算奖励
        rew, done, info = self.get_reward(obs)
        # 回合终止
        truncated = self.time_steps >= self.max_episode_steps
        if truncated or done:
            info["terminal"] = True
            self.__reset_flag = True
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
        traj = pl.array(self.map.start_pos + obs.tolist() + self.map.end_pos) # [x,y,x,y,x,y,...]
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
            d += pl.linalg.norm(vec) 
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

        rew = -d -dθall/self.num_pos -num_theta -num_crash -num_over

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
        """绘图, 必须放step前面"""
        assert not self.__reset_flag, "调用render前必须先reset"

        if self.__render_flag:
            self.__render_flag = False
            pl.ion()       # 打开交互绘图, 只能开一次

        # 清除原图像
        pl.clf() 
        # 障碍物
        for o in self.map.obstacles:
            plot_polygon(o, facecolor='w', edgecolor='k', add_points=False)
        # 起点终点
        pl.scatter(self.map.start_pos[0], self.map.start_pos[1], s=30, c='k', marker='x', label='起点')
        pl.scatter(self.map.end_pos[0], self.map.end_pos[1], s=30, c='k', marker='o', label='终点')
        # 轨迹
        traj = self.map.start_pos + self.obs.tolist() + self.map.end_pos # [x,y,x,y,x,y,]
        pl.plot(traj[::2], traj[1::2], label='path', color='b')
        #pl.legend(loc='best').set_draggable(True) # 显示图例
        pl.legend(loc='best')
        pl.axis('equal')                          # 平均坐标
        pl.xlabel("x")                            # x轴标签
        pl.ylabel("y")                            # y轴标签
        pl.xlim(self.map.size[0][0], self.map.size[1][0]) # x范围
        pl.ylim(self.map.size[0][1], self.map.size[1][1]) # y范围
        pl.title('Path Planning')                 # 标题
        pl.grid()                                 # 生成网格
        pl.grid(alpha=0.3,ls=':')                 # 改变网格透明度，和网格线的样式
        
        pl.pause(0.001)                           # 暂停0.01秒
        pl.ioff()                                 # 禁用交互模式

    def close(self):
        """关闭绘图"""
        self.__render_flag = True
        pl.close()







# 动态环境
from lidar_sim import LidarModel
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

    # 地图设置   ↓↓↓↓ 下述参数设置均与地图尺寸有关 ↓↓↓↓
    MAP = Map()
    # 动力学状态区间设置
    STATE_LOW = [-10, -10, 0.05, -math.pi] # x, z, V, ψ
    STATE_HIGH = [10, 10, 0.2, math.pi]    # x, z, V, ψ
    # 观测状态区间设置
    OBS_STATE_LOW = [0, -math.pi, 0, -math.pi, 0.05, -math.pi]      # 相对起点距离 + 相对起点方位角 rad + 相对终点距离 + 相对终点方位角 rad + 速度 + 偏角
    OBS_STATE_HIGH = [14.14, math.pi, 14.14, math.pi, 0.2, math.pi] # 相对起点距离 + 相对起点方位角 rad + 相对终点距离 + 相对终点方位角 rad + 速度 + 偏角
    # 控制区间设置
    CTRL_LOW = [-0.02, -math.pi/3] # 切向过载 + 速度滚转角 rad
    CTRL_HIGH = [0.02, math.pi/3]  # 切向过载 + 速度滚转角 rad
    # 雷达扫描设置
    SCAN_RANGE = 5   # 扫描距离
    SCAN_ANGLE = 128 # 扫描范围 deg
    SCAN_NUM = 64    # 扫描点个数

    def __init__(self, max_episode_steps=200, safe_radius=0.5, dt=0.5, use_old_gym=True):
        """最大回合步数, 安全距离, 积分步长, 是否采用老版接口
        """
        # 仿真
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.map = self.MAP
        self.obstacles = self.map.obstacles
        self.obstacles_buf = [o.buffer(safe_radius) for o in self.obstacles]
        self.lidar = LidarModel(self.SCAN_RANGE, self.SCAN_ANGLE, self.SCAN_NUM)
        self.lidar.add_obstacles(self.obstacles)

        # 状态空间 + 控制空间
        self.state_space = spaces.Box(pl.array(self.STATE_LOW), pl.array(self.STATE_HIGH))
        self.control_space = spaces.Box(pl.array(self.CTRL_LOW), pl.array(self.CTRL_HIGH))

        # 观测空间 + 动作空间
        obs_point = spaces.Box(0, pl.array(self.SCAN_RANGE), (4, self.SCAN_NUM, )) # batch, seq_len, dim
        obs_state = spaces.Box(pl.array(self.OBS_STATE_LOW), pl.array(self.OBS_STATE_HIGH))
        self.observation_space = spaces.Dict({'point': obs_point, 'state': obs_state})
        self.action_space = spaces.Box(-1, 1, (len(self.CTRL_LOW), ))
        
        self.__render_flag = True
        self.__reset_flag = True
        self.__old_gym = use_old_gym

    def reset(self, mode=0):
        """mode=0, 随机初始化位置
           mode=1, 初始化位置到起点
        """
        self.__reset_flag = False
        obs, info = None, None
        if self.__old_gym:
            return obs
        return obs, info
    
    def step(self, act):
        obs, rew, done, truncated, info = None, None, None, None, None

        if self.__old_gym:
            return obs, rew, done, info
        return obs, rew, done, truncated, info
    
    def render(self, mode="human"):
        pass

    def close(self):
        pass

    

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
        x = pl.linalg.norm(x_vec) * pl.linalg.norm(y_vec)
        if x < EPS:
            return math.pi/2
        return math.acos(pl.clip(pl.dot(x_vec, y_vec) / x, -1, 1)) # note: x很小的时候, 可能会超过+-1
    
    @staticmethod
    def _compute_azimuth(pos1, pos2, use_3d_pos=False):
        """计算pos2相对pos1的方位角 [-π, π] 和高度角(3D情况) [-π/2, π/2] """
        if use_3d_pos:
            x, y, z = pl.array(pos2) - pos1
            q = math.atan(y / (math.sqrt(x**2 + z**2) + 1e-8)) # 高度角 [-π/2, π/2]
            ε = math.atan2(-z, x)                              # 方位角 [-π, π]
            return ε, q
        else:
            x, z = pl.array(pos2) - pos1
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
        return pl.array(s_new[-1]) # deepcopy

  
    
















# 环境-算法适配
class NormalizedActionsWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(NormalizedActionsWrapper, self).__init__(env)
        assert isinstance(env.action_space, spaces.Box), '只用于Box动作空间'
  
    # 将神经网络输出转换成gym输入形式
    def action(self, action): 
        # 连续情况 scale action [-1, 1] -> [lb, ub]
        lb, ub = self.action_space.low, self.action_space.high
        action = lb + (action + 1.0) * 0.5 * (ub - lb)
        action = pl.clip(action, lb, ub)
        return action

    # 将gym输入形式转换成神经网络输出形式
    def reverse_action(self, action):
        # 连续情况 normalized action [lb, ub] -> [-1, 1]
        lb, ub = self.action_space.low, self.action_space.high
        action = 2 * (action - lb) / (ub - lb) - 1
        return pl.clip(action, -1.0, 1.0)
       


if __name__ == '__main__':
    Map.show()