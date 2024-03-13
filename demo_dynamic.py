# -*- coding: utf-8 -*-
"""
静态路径规划示例
 Created on Wed Mar 13 2024 18:18:07
 Modified on 2024-3-13 18:18:07
 
 @auther: HJ https://github.com/zhaohaojie1998
"""

from copy import deepcopy
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.close('all')

import torch as th
import numpy as np
import torch.nn as nn
from copy import deepcopy



'''策略定义'''
class EncoderNet(nn.Module):
    def __init__(self, obs_space, feature_dim):
        super(EncoderNet, self).__init__()
        point_shape = obs_space['point'].shape
        state_dim = obs_space['state'].shape[0]
        self.cnn = nn.Sequential(

        )
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, feature_dim),
            nn.ReLU(True)
        )
        self.fusion = nn.Sequential(
            nn.Linear(2*feature_dim, feature_dim),
            nn.ReLU(True)
        )
    def forward(self, obs):
        f1 = self.cnn(obs['point'])
        f2 = self.mlp(obs['state'])
        return self.fusion(th.cat([f1, f2], -1))
    
class PNet(nn.Module):
    def __init__(self, feature_dim, act_dim):
        super(PNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, act_dim)
        )
    def forward(self, feature):
        return self.mlp(feature)


'''环境实例化'''
from env import DynamicPathPlanning, NormalizedActionsWrapper
env = NormalizedActionsWrapper(DynamicPathPlanning())
obs_space = env.observation_space
act_dim = env.action_space.shape[0]


'''策略实例化'''
from sac_agent import SAC_Actor
policy = SAC_Actor(
        EncoderNet(obs_space, 256),
        PNet(256, act_dim),
        PNet(256, act_dim),
    )
policy.load_state_dict(th.load("demo_dynamic.pkl", map_location="cpu"))


    
'''仿真LOOP'''
MAX_EPISODE = 20        # 总的训练/评估次数
render = True           # 是否可视化训练/评估过程(仿真速度会降几百倍)

for episode in range(MAX_EPISODE):
    ## 获取初始观测
    obs = env.reset()
    ## 进行一回合仿真
    for steps in range(env.max_episode_steps):
        # 可视化
        if render:
            env.render()
        # 决策
        obs_tensor = {
                'point': th.FloatTensor(obs['point']).unsqueeze(0).to("cpu"),
                'state': th.FloatTensor(obs['state']).unsqueeze(0).to("cpu"),
            }
        act = policy.act(obs_tensor)
        # 仿真
        next_obs, _, _, info = env.step(act)
        # 回合结束
        if info["terminal"]:
            print('回合: ', episode,'| 状态: ', info,'| 步数: ', steps) 
            break
        else:
            obs = deepcopy(next_obs)
    #end for
#end for






r'''
#             ⠰⢷⢿⠄
#         ⠀⠀⠀⠀⠀⣼⣷⣄
#         ⠀⠀⣤⣿⣇⣿⣿⣧⣿⡄
#         ⢴⠾⠋⠀⠀⠻⣿⣷⣿⣿⡀
#         🏀   ⢀⣿⣿⡿⢿⠈⣿
#          ⠀⠀⢠⣿⡿⠁⢠⣿⡊⠀⠙
#          ⠀⠀⢿⣿⠀⠀⠹⣿
#           ⠀⠀⠹⣷⡀⠀⣿⡄
#            ⠀⣀⣼⣿⠀⢈⣧ 
#
#       你。。。干。。。嘛。。。
#       哈哈。。唉哟。。。
'''