# -*- coding: utf-8 -*-
"""
策略训练示例
 Created on Sat Nov 04 2023 15:37:28
 Modified on 2023-11-4 15:37:28
 
 @auther: HJ https://github.com/zhaohaojie1998
"""
#


'''算法定义'''
import numpy as np
import torch as th
import torch.nn as nn
from copy import deepcopy
from sac_agent import *
# 1.定义经验回放（取决于观测和动作数据结构）
class Buffer(BaseBuffer):
    def __init__(self, memory_size, obs_space, act_space):
        super(Buffer, self).__init__()
        # 数据类型表示
        self.device = 'cuda'
        self.obs_space = obs_space
        self.act_space = act_space
        # buffer属性
        self._ptr = 0    # 当前位置
        self._idxs = [0] # PER记住上一次采样位置, 一维list或ndarray
        self._memory_size = int(memory_size) # 总容量
        self._current_size = 0               # 当前容量
        # buffer容器
        obs_shape = obs_space.shape or (1, )
        act_shape = act_space.shape or (1, ) # NOTE DiscreteSpace的shape为(), 设置collections应为(1, )
        self._data = {}
        self._data["s"] = np.empty((memory_size, *obs_shape), dtype=obs_space.dtype) # (size, *obs_shape, )连续 (size, 1)离散
        self._data["s_"] = deepcopy(self._data["s"])                                 # (size, *obs_shape, )连续 (size, 1)离散
        self._data["a"] = np.empty((memory_size, *act_shape), dtype=act_space.dtype) # (size, *act_shape, )连续 (size, 1)离散
        self._data["r"] = np.empty((memory_size, 1), dtype=np.float32)               # (size, 1)
        self._data["done"] = np.empty((memory_size, 1), dtype=bool)                  # (size, 1) 

    def reset(self, *args, **kwargs):
        self._ptr = 0
        self._idxs = [0]
        self._current_size = 0

    @property
    def nbytes(self):
        return sum(x.nbytes for x in self._data.values())

    def push(self, transition, terminal=None, **kwargs):
        self._data["s"][self._ptr] = transition[0]
        self._data["a"][self._ptr] = transition[1]
        self._data["r"][self._ptr] = transition[2]
        self._data["s_"][self._ptr] = transition[3]
        self._data["done"][self._ptr] = transition[4]
        # update
        self._ptr = (self._ptr + 1) % self._memory_size                     # 更新指针
        self._current_size = min(self._current_size + 1, self._memory_size) # 更新容量

    def __len__(self):
        return self._current_size 
    
    def sample(self, batch_size=1, *, idxs=None, rate=None, **kwargs):
        self._idxs = idxs or np.random.choice(self._current_size, size=batch_size, replace=False)
        batch = {k: th.FloatTensor(self._data[k][self._idxs]).to(self.device) for k in self._data.keys()}
        return batch
    
    def state_to_tensor(self, state, use_rnn=False):
        return th.FloatTensor(state).unsqueeze(0).to(self.device)
    
# 2.定义神经网络（取决于观测数据结构）
# Q网络
QEncoderNet = nn.Identity

class QNet(nn.Module):
    def __init__(self, feature_dim, act_dim):
        super(QNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim+act_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 1),
        )
    def forward(self, feature_and_action):
        return self.mlp(feature_and_action)

# Pi网络
class PiEncoderNet(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super(PiEncoderNet, self).__init__()
        obs_dim = np.prod(obs_shape)
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, feature_dim),
            nn.ReLU(True),
        )
    def forward(self, obs):
        return self.mlp(obs)
    
class PiNet(nn.Module):
    def __init__(self, feature_dim, act_dim):
        super(PiNet, self).__init__()
        self.mlp = nn.Linear(feature_dim, act_dim)
    def forward(self, feature):
        return self.mlp(feature)



'''实例化环境'''
from path_plan_env import StaticPathPlanning, NormalizedActionsWrapper
env = NormalizedActionsWrapper(StaticPathPlanning())
obs_space = env.observation_space
act_space = env.action_space



'''实例化算法'''
# 1.缓存设置
buffer = Buffer(10000, obs_space, act_space)

# 2.神经网络设置
actor = SAC_Actor(
        PiEncoderNet(obs_space.shape, 128),
        PiNet(128, act_space.shape[0]),
        PiNet(128, act_space.shape[0]),
    )
critic = SAC_Critic(
        QEncoderNet(),
        QNet(obs_space.shape[0], act_space.shape[0]),
        QNet(obs_space.shape[0], act_space.shape[0]),
    )

# 3.算法设置
agent = SAC_Agent(env)
agent.set_buffer(buffer)
agent.set_nn(actor, critic)
agent.cuda()



'''训练LOOP''' 
MAX_EPISODE = 2000
for episode in range(MAX_EPISODE):
    ## 重置回合奖励
    ep_reward = 0
    ## 获取初始观测
    obs = env.reset()
    ## 进行一回合仿真
    for steps in range(env.max_episode_steps):
        # 决策
        act = agent.select_action(obs)
        # 仿真
        next_obs, reward, done, info = env.step(act)
        ep_reward += reward
        # 缓存
        agent.store_memory((obs, act, reward, next_obs, done))
        # 优化
        agent.learn()
        # 回合结束
        if info["terminal"]:
            mean_reward = ep_reward / (steps + 1)
            print('回合: ', episode,'| 累积奖励: ', round(ep_reward, 2),'| 平均奖励: ', round(mean_reward, 2),'| 状态: ', info,'| 步数: ', steps) 
            break
        else:
            obs = deepcopy(next_obs)
    #end for
#end for
agent.export("./path_plan_env/policy_static.onnx") # 导出策略模型
# agent.save("./checkpoint") # 存储算法训练进度
# agent.load("./checkpoint") # 加载算法训练进度







r'''
                           _ooOoo_
                          o8888888o
                          88" . "88
                          (| -_- |)
                          O\  =  /O
                       ____/`---'\____
                     .'  \\|     |//  `.
                    /  \\|||  :  |||//  \
                   /  _||||| -:- |||||-  \
                   |   | \\\  -  /// |   |
                   | \_|  ''\---/''  |   |
                   \  .-\__  `-`  ___/-. /
                 ___`. .'  /--.--\  `. . __
              ."" '<  `.___\_<|>_/___.'  >'"".
             | | :  `- \`.;`\ _ /`;.`/ - ` : | |
             \  \ `-.   \_ __\ /__ _/   .-` /  /
        ======`-.____`-.___\_____/___.-`____.-'======
                           `=---='
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    佛祖保佑       永无BUG
'''
