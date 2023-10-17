# -*- coding: utf-8 -*-
"""
 Created on Fri Mar 03 2023 19:58:10
 Modified on 2023-3-3 19:58:
     
 @auther: HJ https://github.com/zhaohaojie1998
"""

# 阉割版代码
# 完整版DRL-Algorithm(暂时不开源)包含多种DRL/MADRL算法, 有文件系统, 绘图系统, 训练日志系统, 万能Buffer, 万能数据转换模块等


# Runing on GPU #
import gym
import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal
from copy import deepcopy



LOG_STD_MAX = 2
LOG_STD_MIN = -20



# 创建MLP模型
def build_mlp(layer_shape, activation=nn.ReLU, output_activation=nn.Identity, inplace=True):
    """创建MLP模型

    Parameters
    ----------
    layer_shape : MLP形状, list / tuple
    activation : MLP激活函数, 默认 nn.ReLU
    output_activation : MLP输出激活函数, 默认 nn.Identity
    inplace : 例如ReLU之类的激活函数是否设置inplace, 默认 True

    """
    def _need_inplace(activation) -> bool:
        return activation == nn.ReLU or\
                activation == nn.ReLU6 or\
                activation == nn.RReLU or\
                activation == nn.LeakyReLU or\
                activation == nn.SiLU or\
                activation == nn.ELU or\
                activation == nn.SELU or\
                activation == nn.CELU or\
                activation == nn.Threshold or\
                activation == nn.Hardsigmoid or\
                activation == nn.Hardswish or\
                activation == nn.Mish
    layers = []
    for j in range(len(layer_shape)-1):
        act = activation if j < len(layer_shape)-2 else output_activation
        if inplace and _need_inplace(act):
            layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act(inplace=True)] # 加快速度, 减小显存占用
        else:
            layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()] # 例如tanh没有inplace参数
    return nn.Sequential(*layers)


# Q网络
class Q_Critic(nn.Module):
    def __init__(self, num_states:int, num_actions:int, hidden_dims:tuple=(128,64)):
        super(Q_Critic, self).__init__()
        mlp_shape = [num_states + num_actions] + list(hidden_dims) + [1]
        self.Q1_Value = build_mlp(mlp_shape)
        self.Q2_Value = build_mlp(mlp_shape)

    def forward(self, obs, action):
        obs = nn.Flatten()(obs)
        x = th.cat([obs, action], -1)
        Q1 = self.Q1_Value(x)
        Q2 = self.Q2_Value(x)
        return Q1, Q2


# P网络(OpenAI版)
class Actor(nn.Module):
    def __init__(self, num_states:int, num_actions:int, hidden_dims=(128,128)):
        super(Actor, self).__init__()
        layer_shape = [num_states] + list(hidden_dims)
        self.mlp_layer = build_mlp(layer_shape, output_activation=nn.ReLU)
        self.mu_layer = nn.Linear(layer_shape[-1], num_actions)
        self.log_std_layer = nn.Linear(layer_shape[-1], num_actions)

        self.LOG_STD_MAX = LOG_STD_MAX
        self.LOG_STD_MIN = LOG_STD_MIN

    def forward(self, obs, deterministic=False, with_logprob=True):
        obs = nn.Flatten()(obs)
        x = self.mlp_layer(obs)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX) # ???
        std = th.exp(log_std)

        # 策略分布
        dist = Normal(mu, std)
        if deterministic: u = mu
        else: u = dist.rsample() # 重参数化采样(直接采样不可导)

        a = th.tanh(u) # 压缩输出范围[-1, 1]

        # 计算正态分布概率的对数 log[P_pi(a|s)] -> (batch, act_dim)
        if with_logprob:
            # SAC论文通过u的对数概率计算a的对数概率公式:
            # logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)

            # SAC原文公式有a=tanh(u), 导致梯度消失, OpenAI公式:
            logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True) # (batch, 1)
        else:
            logp_pi_a = None

        return a, logp_pi_a # (batch, act_dim) and (batch, 1)

    def act(self, obs, deterministic=False) -> np.ndarray[any, float]:
        self.eval()
        with th.no_grad():
            a, _ = self.forward(obs, deterministic, False)
        self.train()
        return a.cpu().numpy().flatten() # (act_dim, ) ndarray
    




# ReplayBuffer
class EasyBuffer:
    """阉割版缓存"""

    def __init__(self, memory_size, obs_space, act_space):
        assert not isinstance(obs_space, (gym.spaces.Tuple, gym.spaces.Dict)), "阉割版缓存只能存简单数据"
        assert not isinstance(act_space, (gym.spaces.Tuple, gym.spaces.Dict)), "阉割版缓存只能存简单数据"
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        # buffer属性
        self.ptr = 0    # buffer存储指针
        self.idxs = [0] # 用于PER记住上一次采样索引, 一维 list 或 ndarray
        self.memory_size = memory_size
        self.current_size = 0 
        # buffer容器
        obs_shape = obs_space.shape or (1, )
        act_shape = obs_space.shape or (1, )
        self.buffer = {}
        self.buffer["obs"] = np.empty((memory_size, *obs_shape), dtype=obs_space.dtype) # (size, *obs_shape, )连续 (size, 1)离散
        self.buffer["next_obs"] = deepcopy(self.buffer["obs"])                          # (size, *obs_shape, )连续 (size, 1)离散
        self.buffer["act"] = np.empty((memory_size, *act_shape), dtype=act_space.dtype) # (size, *act_shape, )连续 (size, 1)离散
        self.buffer["rew"] = np.empty((memory_size, 1), dtype=np.float32)               # (size, 1)
        self.buffer["done"] = np.empty((memory_size, 1), dtype=bool)                    # (size, 1) 

    def __getitem__(self, position):
        """索引\n
        即 batch = buffer[position] 与 batch = buffer.sample(idxs=position) 效果相同
        """
        if isinstance(position, int): position = [position]
        return self.sample(idxs=position)

    def __len__(self):
        return self.current_size 

    def reset(self):
        """清空"""
        self.ptr = 0
        self.idxs = [0]
        self.current_size = 0

    def push(self, state, action, reward, next_state, done, terminal=None):
        """存储"""
        # add a transition to the buffer
        self.buffer["obs"][self.ptr] = state
        self.buffer["next_obs"][self.ptr] = next_state
        self.buffer["act"][self.ptr] = action
        self.buffer["rew"][self.ptr] = reward
        self.buffer["done"][self.ptr] = done
        # update ptr & size
        self.ptr = (self.ptr + 1) % self.memory_size # 更新指针
        self.current_size = min(self.current_size + 1, self.memory_size) # 更新容量

    def sample(self, batch_size = 1, *, idxs = None, rate = None, **kwargs):
        """采样"""
        # make indexes
        if idxs is None:
            assert batch_size <= self.current_size, "batch_size 要比当前容量小"
            idxs = np.random.choice(self.current_size, size=batch_size, replace=False)
        # sample a batch from the buffer
        batch = {}
        for key in self.buffer:
            if key != "act":
                batch[key] = th.FloatTensor(self.buffer[key][idxs]).to(self.device)
            else:
                batch[key] = th.tensor(self.buffer[key][idxs]).to(self.device)
        # update indexes
        self.idxs = idxs
        return batch

        
        
    

# SAC 1812 Agent
# 论文:《Soft Actor-Critic Algorithms and Applications》
class SAC:
    """Soft Actor-Critic (arXiv: 1812)"""
   
    def __init__(
        self, 
        observation_space: gym.Space, # 观测空间
        action_space: gym.Space,      # 动作空间

        *,
        memory_size: int = 100000,  # 缓存大小
            
        gamma: float = 0.99,        # 折扣因子 γ
        alpha: float = 0.2,         # 温度系数 α
        
        batch_size: int = 128,      # 样本容量
        update_after: int = 1000,   # 训练开始，batch_size <= update_after <= memory_size

        lr_decay_period: int = None, # 学习率衰减周期, None不衰减
        lr_critic: float = 1e-3,     # Q 学习率
        lr_actor: float = 1e-3,      # π 学习率
        tau: float = 0.005,          # target Q 软更新系数 τ

        q_loss_cls = nn.MSELoss,  # Q 损失函数类型(use_per=True时该设置无效)
        
        critic_optim_cls = th.optim.Adam, # Q 优化器类型
        actor_optim_cls = th.optim.Adam,  # π 优化器类型
        
        adaptive_alpha: bool = True,       # 是否自适应温度系数
        target_entropy: float = None,      # 自适应温度系数目标熵, 默认: -dim(A)
        lr_alpha: float = 1e-3,            # α 学习率
        alpha_optim_class = th.optim.Adam, # α 优化器类型

        use_per: bool = False,  # 是否优先经验回放
        per_alpha: float = 0.6, # 优先回放 α
        per_beta0: float = 0.4, # 优先回放 β

        grad_clip: float = None, # Q网络梯度裁剪范围, None不裁剪

    ):
        assert isinstance(action_space, gym.spaces.Box), 'SAC只用于Box动作空间'
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

        # 环境参数
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_states = np.prod(observation_space.shape)
        self.num_actions = np.prod(action_space.shape)

        # SAC参数初始化
        self.gamma = gamma
        self.batch_size = int(batch_size)
        self.update_after = int(update_after)

        self.lr_decay_period = lr_decay_period
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.tau = tau

        # ReplayBuffer初始化
        self.memory_size = int(memory_size)
        self.use_per = use_per
        if use_per:
            raise NotImplementedError("阉割版代码不支持PER")
        else:
            self.buffer = EasyBuffer(self.memory_size, self.observation_space, self.action_space)

        # 神经网络初始化
        self.actor = Actor(self.num_states, self.num_actions).to(self.device)
        self.q_critic = Q_Critic(self.num_states, self.num_actions).to(self.device) # Twin Q Critic
        self.target_q_critic = self._build_target(self.q_critic)
        
        # 优化器初始化
        self.actor_optimizer = actor_optim_cls(self.actor.parameters(), lr_actor)
        self.q_critic_optimizer = critic_optim_cls(self.q_critic.parameters(), lr_critic)
        
        # 设置损失函数
        self.grad_clip = grad_clip
        self.q_loss = q_loss_cls()
        
        # 是否自适应α
        self.alpha = alpha
        self.adaptive_alpha = adaptive_alpha
        if adaptive_alpha:
            target_entropy = target_entropy or -self.num_actions # Target Entropy = −dim(A)
            self.target_entropy = th.tensor(target_entropy, dtype=float, requires_grad=True, device=self.device)
            self.log_alpha = th.tensor(np.log(alpha), dtype=float, requires_grad=True, device=self.device) # log_alpha无>0的约束
            self.alpha_optimizer = alpha_optim_class([self.log_alpha], lr = lr_alpha)
            self.lr_alpha = lr_alpha

        # 其它参数
        self.learn_counter = 0
    

    def setup_nn(self, actor: Actor, critic: Q_Critic, *, actor_optim_cls=th.optim.Adam, critic_optim_cls=th.optim.Adam, copy=True):
        """修改神经网络模型, 要求按Actor/Q_Critic格式自定义网络"""
        self.actor = deepcopy(actor) if copy else actor
        self.actor.train().to(self.device)
        self.q_critic = deepcopy(critic) if copy else critic
        self.q_critic.train().to(self.device) # Twin Q Critic
        self.target_q_critic = self._build_target(self.q_critic)
        self.actor_optimizer = actor_optim_cls(self.actor.parameters(), self.lr_actor)
        self.q_critic_optimizer = critic_optim_cls(self.q_critic.parameters(), self.lr_critic)


    def store_memory(self, s, a, r, s_, d):
        """经验存储"""
        self.buffer.push(s, a, r, s_, d)


    def select_action(self, state, *, deterministic=False, **kwargs) -> np.ndarray:
        """选择动作"""
        state = th.FloatTensor(state).unsqueeze(0).to(self.device) # (1, state_dim) tensor
        return self.actor.act(state, deterministic) # (act_dim, ) ndarray


    def learn(self, *, rate: float = None) -> dict:
        """强化学习

        Parameters
        ----------
        rate : float, optional
            用于更新PER的参数 beta, 默认None不更新
            rate = train_steps / max_train_steps
            beta = beta0 + (1-beta0) * rate
        """
      
        if len(self.buffer) < self.batch_size or \
            len(self.buffer) < self.update_after:    
            return {'q_loss': None, 'actor_loss': None, 'alpha_loss': None, 'q': None, 'alpha': None}
        
        self.learn_counter += 1
        
        ''' experience repaly '''
        samples = self.buffer.sample(self.batch_size, rate=rate) # return tensor GPU
        state = samples["obs"]           # (m, obs_dim) tensor GPU
        action = samples["act"]          # (m, act_dim) tensor GPU
        reward = samples["rew"]          # (m, 1) tensor GPU
        next_state = samples["next_obs"] # (m, obs_dim) tensor GPU
        done = samples["done"]           # (m, 1) tensor GPU
        if self.use_per:
            IS_weight = samples["IS_weight"] # (m, 1) tensor GPU


        ''' Q Critic 网络优化 '''
        # J(Q) = E_{s_t~D, a_t~D, s_t+1~D, a_t+1~π_t+1}[0.5*[ Q(s_t, a_t) - [r + (1-d)*γ* [ Q_tag(s_t+1,a_t+1) - α*logπ_t+1 ] ]^2 ]
        # 计算目标 Q 值
        with th.no_grad():
            next_action, next_log_pi = self.actor(next_state)                                 # (m, act_dim), (m, 1) tensor GPU no grad
            Q1_next, Q2_next = self.target_q_critic(next_state, next_action)                  # (m, 1) tensor GPU no grad
            Q_next = th.min(Q1_next, Q2_next)                                                 # (m, 1) tensor GPU no grad
            #V_next = Q_next - self.alpha * next_log_pi
            Q_targ = reward + (1.0 - done) * self.gamma * (Q_next - self.alpha * next_log_pi) # (m, 1) tensor GPU no grad

        # 计算当前 Q 值
        Q1_curr, Q2_curr = self.q_critic(state, action) # (m, 1) tensor GPU with grad

        # 计算损失
        if self.use_per:
            td_err1, td_err2 = Q1_curr-Q_targ, Q2_curr-Q_targ  # (m, 1) tensor GPU with grad
            q_loss = (IS_weight * (td_err1 ** 2)).mean() + (IS_weight * (td_err2 ** 2)).mean() # () 注意: mean一定加在最外面！！！！
            self.buffer.update_priorities(td_err1.detach().cpu().numpy().flatten()) # 更新优先级 td err: (m, ) ndarray
        else:
            q_loss = self.q_loss(Q1_curr, Q_targ) + self.q_loss(Q2_curr, Q_targ) # ()

        # 优化网络
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.q_critic.parameters(), self.grad_clip)
        self.q_critic_optimizer.step()


        ''' Actor 网络优化 '''
        # J(π) = E_{s_t~D, a~π_t}[ α*logπ_t(a|π_t) - Q(s_t, a) ]   
        self._freeze_network(self.q_critic)
      
        # 策略评估
        new_action, log_pi = self.actor(state)    # (m, act_dim), (m, 1) tensor GPU with grad
        Q1, Q2 = self.q_critic(state, new_action) # (m, 1) tensor GPU no grad
        Q = th.min(Q1, Q2)                        # (m, 1) tensor GPU no grad

        # 策略优化
        a_loss = (self.alpha * log_pi - Q).mean()
        self._optim_step(self.actor_optimizer, a_loss)

        self._unfreeze_network(self.q_critic)


        ''' alpha 温度系数优化 '''
        # J(α) = E_{a~π_t}[ -α * ( logπ_t(a|π_t) + H0 ) ]
        if self.adaptive_alpha:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()      # log公式: 收敛较快, 计算较快
            #alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean() # 原公式: 收敛用的episode较大, 且计算速度慢
            self._optim_step(self.alpha_optimizer, alpha_loss)
            self.alpha = self.log_alpha.exp().item()
            alpha_loss_scalar = alpha_loss.item()
        else:
            alpha_loss_scalar = None


        ''' target Q 网络优化'''
        self._soft_update(self.target_q_critic, self.q_critic, self.tau)
        

        ''' use lr decay '''
        self._lr_decay(self.actor_optimizer)
        self._lr_decay(self.q_critic_optimizer)
        if self.adaptive_alpha:
            self._lr_decay(self.alpha_optimizer)


        ''' return info '''
        return {'q_loss': q_loss.item(), 
                'actor_loss': a_loss.item(), 
                'alpha_loss': alpha_loss_scalar, 
                'q': Q1_curr.mean().item(), 
                'alpha': self.alpha
                }
    

    def save(self, file):
        """存储Actor网络权重"""
        th.save(self.actor.state_dict(), file)
        
    
    def load(self, file):
        """加载Actor网络权重"""
        self.actor.load_state_dict(th.load(file, map_location=self.device))

    
    @staticmethod
    def _soft_update(target_network: nn.Module, network: nn.Module, tau: float):
        """
        目标神经网络软更新\n
        >>> for target_param, param in zip(target_network.parameters(), network.parameters()):
        >>>    target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        """
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_( target_param.data * (1.0 - tau) + param.data * tau ) # 软更新
   
    @staticmethod
    def _hard_update(target_network: nn.Module, network: nn.Module):
        """
        目标神经网络硬更新\n
        >>> target_network.load_state_dict(network.state_dict())
        """
        # for target_param, param in zip(target_network.parameters(), network.parameters()):
        #     target_param.data.copy_(param.data)
        target_network.load_state_dict(network.state_dict()) # 硬更新

    @staticmethod
    def _freeze_network(network: nn.Module):
        """
        冻结神经网络\n
        >>> for p in network.parameters():
        >>>     p.requires_grad = False
        """
        for p in network.parameters():
            p.requires_grad = False
        # requires_grad = False 用于串联网络(梯度需要传回)计算损失, 如将Q梯度传回Actor网络, 但不更新Critic
        # with th.no_grad() 用于并联或串联网络(梯度不需要传回)计算损失, 如Actor计算next_a, 用next_a计算Q, 但Q梯度不传回Actor

    @staticmethod
    def _unfreeze_network(network: nn.Module):
        """
        解冻神经网络\n
        >>> for p in network.parameters():
        >>>     p.requires_grad = True
        """
        for p in network.parameters():
            p.requires_grad = True

    @staticmethod
    def _build_target(network: nn.Module):
        """
        拷贝一份目标网络\n
        >>> target_network = deepcopy(network).eval()
        >>> for p in target_network.parameters():
        >>>     p.requires_grad = False
        """
        target_network = deepcopy(network).eval()
        for p in target_network.parameters():
            p.requires_grad = False
        return target_network
    
    @staticmethod
    def _set_lr(optimizer: th.optim.Optimizer, lr: float):
        """
        调整优化器学习率\n
        >>> for g in optimizer.param_groups:
        >>>     g['lr'] = lr
        """
        for g in optimizer.param_groups:
            g['lr'] = lr

    def _lr_decay(self, optimizer: th.optim.Optimizer):
        """学习率衰减 (在 lr_decay_period 周期内衰减到初始的 0.1 倍, period 为 None/0 不衰减)
        >>> lr = 0.9 * lr_init * max(0, 1 - step / lr_decay_period) + 0.1 * lr_init
        >>> self._set_lr(optimizer, lr)
        """
        if self.lr_decay_period:
            lr_init = optimizer.defaults["lr"] # 读取优化器初始化时的 lr_init
            lr = 0.9 * lr_init * max(0, 1 - self.learn_counter / self.lr_decay_period) + 0.1 * lr_init # 更新 lr
            self._set_lr(optimizer, lr) # 更改 param_groups 的 lr
            # NOTE 修改 param_groups 的 lr 并不会改变 defaults 的 lr

    @staticmethod
    def _optim_step(optimizer: th.optim.Optimizer, loss: th.Tensor):
        """
        神经网络权重更新\n
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
        """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

