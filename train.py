# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 15:45:25 2021

@author: ZHAOHAOJIE
"""
#

'''训练程序'''
import pylab as pl
from copy import deepcopy
from env import PathPlanning, NormalizedActionsWrapper

from sac import SAC



pl.mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 修复字体bug
pl.close('all')                                          # 关闭所有窗口


'''模式设置''' 
MAX_EPISODE = 2000        # 总的训练/评估次数
render = False           # 是否可视化训练/评估过程(仿真速度会降几百倍)


'''环境算法设置'''
env = PathPlanning()
env = NormalizedActionsWrapper(env)
agent = SAC(env.observation_space, env.action_space, memory_size=10000) # 实例化强化学习算法


    
'''强化学习训练/测试仿真'''
for episode in range(MAX_EPISODE):
    ## 重置回合奖励
    ep_reward = 0
    
    ## 获取初始观测
    obs = env.reset()
    
    ## 进行一回合仿真
    for steps in range(env.max_episode_steps):
        # 可视化
        if render:
            env.render()
        
        # 决策
        act = agent.select_action(obs)  # 随机策略

        # 仿真
        next_obs, reward, done, info = env.step(act)
        ep_reward += reward
        
        # 缓存
        agent.store_memory(obs, act, reward, next_obs, done)
        
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
agent.save()





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



