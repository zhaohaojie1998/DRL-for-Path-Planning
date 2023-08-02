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





class Map:
    size = [[-10.0, -10.0], [10.0, 10.0]] # x, y最小值; x, y最大值
    start_pos = [0, -9]           # 起点坐标
    end_pos = [2.5, 9]            # 终点坐标
    obstacle = [                  # 障碍物坐标+半径
        [0, 2.5, 4],
        [-6, -5, 3],
        [6, -5, 3],
    ]

    @classmethod
    def show(cls):
        pl.mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 修复字体bug
        pl.mpl.rcParams['axes.unicode_minus'] = False            # 用来正常显示负号

        pl.close('all')
        pl.figure('Map')
        pl.clf()
    
        # 障碍物
        for o in cls.obstacle:
            circle = pl.Circle((o[0], o[1]), o[2], color='k', fill=False)
            pl.gcf().gca().add_artist(circle)
       
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
        pl.grid(alpha=0.3,ls=':')                 # 改变网格透明度，和网格线的样式
        pl.show(block=True)

        


# 静态环境 - 老版gym接口格式
class StaticPathPlanning(gym.Env):
    """从航点搜索的角度进行规划"""

    def __init__(self, num_pos = 6, map_class = Map, max_search_steps = 200):
        """起点终点之间的导航点个数, 地图信息, 最大搜索次数
        """
        self.num_pos = num_pos
        self.map = map_class
        self.max_episode_steps = max_search_steps

        lb = pl.array(self.map.size[0] * num_pos)
        ub = pl.array(self.map.size[1] * num_pos)
        self.observation_space = spaces.Box(lb, ub, dtype=pl.float32)
        self.action_space = spaces.Box(lb/10, ub/10, dtype=pl.float32)

        self.__render_flag = True
        self.__reset_flag = True

    def reset(self):
        self.__reset_flag = False
        self.time_steps = 0 # NOTE: 容易简写成step, 会和step method重名, 还不容易发现 BUG
        self.obs = self.observation_space.sample()
        return self.obs
    
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
        if self.time_steps >= self.max_episode_steps or done:
            info["terminal"] = True
            self.__reset_flag = True
        else:
            info["terminal"] = False

        self.obs = deepcopy(obs)
        return obs, rew, done, info
    
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
            for o in self.map.obstacle:
                P1, P2 = traj[i], traj[i+1]
                T, 半径 = pl.array(o[:2]), o[-1]
                c = pl.linalg.norm(P1 - P2) # 轨迹长度
                a = pl.linalg.norm(T - P2) # 下个点和威胁中心的距离
                b = pl.linalg.norm(P1 - T) # 上个点和威胁中心的距离
                if a < 半径 or b < 半径:
                    num_crash += 1
                elif pl.linalg.norm(pl.cross(T-P1, T-P2)) / (c + 1e-8) < 半径\
                and c > a and c > b:
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
        for o in self.map.obstacle:
            circle = pl.Circle((o[0], o[1]), o[2], color='k', fill=False)
            pl.gcf().gca().add_artist(circle)
       
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







# 动态环境 - 新版gym接口格式
class DynamicPathPlanning(gym.Env):
    """从力学与控制的角度进行规划
    >>> dx/dt = V * cos(θ)
    >>> dy/dt = V * sin(θ)
    >>> dθ/dt = V/L * tan(δ)
    >>> dV/dt = a
    >>> u = [a, δ]
    
    """

    def __init__(self):
        pass

    def reset(self, *args, **kwargs):
        self.__reset_flag = False
        obs, info = None, None
        return obs, info
    
    def step(self, act):
        obs, rew, done, truncated, info = None, None, None, None, None
        return obs, rew, done, truncated, info
    
    def render(self, mode="human"):
        pass

    def close(self):
        pass






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