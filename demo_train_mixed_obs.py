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
        self._data = {}
        self._data["points"] = np.empty((memory_size, *obs_space['seq_points'].shape), dtype=obs_space['seq_points'].dtype)
        self._data["points_"] = deepcopy(self._data["points"])
        self._data["vector"] = np.empty((memory_size, *obs_space['seq_vector'].shape), dtype=obs_space['seq_vector'].dtype)
        self._data["vector_"] = deepcopy(self._data["vector"])
        self._data["a"] = np.empty((memory_size, *act_space.shape), dtype=act_space.dtype)
        self._data["r"] = np.empty((memory_size, 1), dtype=np.float32)
        self._data["done"] = np.empty((memory_size, 1), dtype=bool)
    
    def reset(self, *args, **kwargs):
        self._ptr = 0
        self._idxs = [0]
        self._current_size = 0

    @property
    def nbytes(self):
        return sum(x.nbytes for x in self._data.values())

    def push(self, transition, terminal=None, **kwargs):
        self._data["points"][self._ptr] = transition[0]['seq_points']
        self._data["vector"][self._ptr] = transition[0]['seq_vector']
        self._data["a"][self._ptr] = transition[1]
        self._data["r"][self._ptr] = transition[2]
        self._data["points_"][self._ptr] = transition[3]['seq_points']
        self._data["vector_"][self._ptr] = transition[3]['seq_vector']
        self._data["done"][self._ptr] = transition[4]
        # update
        self._ptr = (self._ptr + 1) % self._memory_size                     # 更新指针
        self._current_size = min(self._current_size + 1, self._memory_size) # 更新容量

    def __len__(self):
        return self._current_size 
    
    def sample(self, batch_size=1, *, idxs=None, rate=None, **kwargs):
        self._idxs = idxs or np.random.choice(self._current_size, size=batch_size, replace=False)
        batch = {
            's': {
                'seq_points': th.FloatTensor(self._data['points'][self._idxs]).to(self.device),
                'seq_vector': th.FloatTensor(self._data['vector'][self._idxs]).to(self.device),
            },
            'a': th.FloatTensor(self._data['a'][self._idxs]).to(self.device),
            'r': th.FloatTensor(self._data['r'][self._idxs]).to(self.device),
            's_': {
                'seq_points': th.FloatTensor(self._data['points_'][self._idxs]).to(self.device),
                'seq_vector': th.FloatTensor(self._data['vector_'][self._idxs]).to(self.device),
            },
            'done': th.FloatTensor(self._data['done'][self._idxs]).to(self.device),
        }
        return batch
    
    def state_to_tensor(self, state, use_rnn=False):
        return {'seq_points': th.FloatTensor(state['seq_points']).unsqueeze(0).to(self.device),
                'seq_vector': th.FloatTensor(state['seq_vector']).unsqueeze(0).to(self.device)}
    

# 2.定义神经网络（取决于观测数据结构）
# 混合观测编码器
class EncoderNet(nn.Module):
    def __init__(self, obs_space, feature_dim):
        super(EncoderNet, self).__init__()
        # 点云测距编码
        c, cnn_dim = obs_space['seq_points'].shape
        in_kernel_size = min(cnn_dim//2, 8)
        in_stride = min(cnn_dim-in_kernel_size, 4)
        self.cnn = nn.Sequential(
            nn.Conv1d(c, 32, kernel_size=in_kernel_size, stride=in_stride, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        cnn_out_dim = self._get_cnn_out_dim(self.cnn, (c, cnn_dim))
        self.cnn_mlp = nn.Sequential(
            nn.Linear(cnn_out_dim, feature_dim),
            nn.ReLU(True),
        )
        # 状态向量编码
        _, rnn_dim = obs_space['seq_vector'].shape
        rnn_hidden_dim = 256
        rnn_num_layers = 1
        self.rnn_mlp1 = nn.Sequential(
            nn.Linear(rnn_dim, rnn_hidden_dim),
            nn.ReLU(True),
        )
        self.rnn = nn.GRU(rnn_hidden_dim, rnn_hidden_dim, rnn_num_layers, batch_first=True)
        self.rnn_mlp2 = nn.Sequential(
            nn.Linear(rnn_hidden_dim, feature_dim),
            nn.ReLU(True),
        )
        # 特征融合网络
        self.fusion = nn.Sequential(
            nn.Linear(2*feature_dim, feature_dim),
            nn.ReLU(True),
        )

    def forward(self, obs):
        f1 = self.cnn_mlp(self.cnn(obs['seq_points'])) # batch, dim
        f2_n, _ = self.rnn(self.rnn_mlp1(obs['seq_vector']), None) # batch, seq, dim
        f2 = self.rnn_mlp2(f2_n[:, -1, :]) # batch, dim
        return self.fusion(th.cat([f1, f2], dim=-1)) # batch, dim
    
    @staticmethod
    def _get_cnn_out_dim(cnn: nn.Module, input_shape: tuple[int, ...]):
        # out_dim = (in_dim + 2*pad - dilation*(k_size-1) -1 ) / stride + 1
        cnn_copy = deepcopy(cnn).to('cpu')
        output = cnn_copy(th.zeros(1, *input_shape))
        return int(np.prod(output.size()))

# Q函数
class QNet(nn.Module):
    def __init__(self, feature_dim, act_dim):
        super(QNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim+act_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
        )

    def forward(self, feature_and_action):
        return self.mlp(feature_and_action)

# 策略函数
class PiNet(nn.Module):
    def __init__(self, feature_dim, act_dim):
        super(PiNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, act_dim),
        )

    def forward(self, feature):
        return self.mlp(feature)



'''实例化环境'''
from path_plan_env import DynamicPathPlanning
env = DynamicPathPlanning()
obs_space = env.observation_space
act_space = env.action_space



'''实例化算法'''
# 1.缓存设置
buffer = Buffer(100000, obs_space, act_space)

# 2.神经网络设置
actor = SAC_Actor(
        EncoderNet(obs_space, 256),
        PiNet(256, act_space.shape[0]),
        PiNet(256, act_space.shape[0]),
    )
critic = SAC_Critic(
        EncoderNet(obs_space, 256),
        QNet(256, act_space.shape[0]),
        QNet(256, act_space.shape[0]),
    )

# 3.算法设置
agent = SAC_Agent(env, batch_size=2048)
agent.set_buffer(buffer)
agent.set_nn(actor, critic)
agent.cuda()



'''训练LOOP'''
from torch.utils.tensorboard import SummaryWriter # TensorBoard, 启动!!!
log = SummaryWriter(log_dir = "./tb_log") 

MAX_EPISODE = 50000
LEARN_FREQ = 100
OUTPUT_FREQ = 50
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
        # 回合结束
        if info["terminal"]:
            mean_reward = ep_reward / (steps + 1)
            print('回合: ', episode,'| 累积奖励: ', round(ep_reward, 2),'| 平均奖励: ', round(mean_reward, 2),'| 状态: ', info,'| 步数: ', steps) 
            break
        else:
            obs = deepcopy(next_obs)
    #end for
    ## 记录
    log.add_scalar('Return', ep_reward, episode)
    log.add_scalar('MeanReward', mean_reward, episode)
    ## 训练
    if episode % LEARN_FREQ == 0:
        train_info = agent.learn()
    if episode % OUTPUT_FREQ == 0:
        env.plot(f"./output/out{episode}.png")
#end for
agent.export("./path_plan_env/policy_dynamic.onnx") # 导出策略模型
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
