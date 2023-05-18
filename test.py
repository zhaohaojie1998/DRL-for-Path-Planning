# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 15:45:25 2021

@author: ZHAOHAOJIE
"""
#


'''测试程序'''
import torch
import pylab as pl
from copy import deepcopy
from env import PathPlanning, NormalizedActionsWrapper

from sac import SAC


pl.mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 修复字体bug
pl.close('all')                                          # 关闭所有窗口


'''模式设置''' 
MAX_EPISODE = 20        # 总的训练/评估次数
render = True           # 是否可视化训练/评估过程(仿真速度会降几百倍)


'''环境算法设置'''
env = PathPlanning()
env = NormalizedActionsWrapper(env)
agent = SAC(env.observation_space, env.action_space, memory_size=10000)
agent.load()


    
'''强化学习训练/测试仿真'''
for episode in range(MAX_EPISODE):
    ## 获取初始观测
    obs = env.reset()
    
    ## 进行一回合仿真
    for steps in range(env.max_episode_steps):
        # 可视化
        if render:
            env.render()
        
        # 决策
        act = agent.select_action(obs)

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