# -*- coding: utf-8 -*-
"""
SAC-Auto算法
 Created on Fri Mar 03 2023 19:58:10
 Modified on 2023-3-3 19:58:
     
 @auther: HJ https://github.com/zhaohaojie1998
"""
# Runing on GPU #
from rl_typing import *
from abc import abstractmethod

import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal

from pathlib import Path
from copy import deepcopy

__all__ = [
    "BaseBuffer",
    "SAC_Critic",
    "SAC_Actor",
    "SAC_Agent",
]






#----------------------------- ↓↓↓↓↓ Experience Replay Buffer ↓↓↓↓↓ ------------------------------#
class BaseBuffer:
    """ReplayBuffer坤类, 需根据具体任务完善相应功能"""

    obs_space: ObsSpace
    act_space: ActSpace
    device: DeviceLike = 'cpu'

    # 0.重置
    @abstractmethod
    def reset(self, *args, **kwargs):
        """重置replay buffer"""
        raise NotImplementedError
    
    @property
    def is_rnn(self) -> bool:
        """是否RNN replay"""
        return False
    
    @property
    def nbytes(self) -> int:
        """buffer占用的内存"""
        return 0
    
    # 1.存储
    @abstractmethod
    def push(
        self, 
        transition: tuple[Obs, Act, float, Obs, bool], 
        terminal: bool = None, 
        **kwargs
    ):
        """存入一条样本\n
            transition = (state, action, reward, next_state, done)
            terminal 用于控制 DRQN 的 EPISODE REPLAY
        """
        raise NotImplementedError
    
    def __len__(self) -> int:
        """当前buffer容量"""
        return 0
    
    def extend(
        self, 
        transition_list: list[tuple[Obs, Act, float, Obs, bool]], 
        terminal_list: list[bool] = None, 
        **kwargs
    ):
        """存入一批样本\n
            extend(List[(state, action, reward, next_state, done)], List[terminal])
        """
        for transition, terminal in zip(transition_list, terminal_list):
            self.push(transition, terminal)

    # 2.采样
    @abstractmethod
    def sample(
        self, 
        batch_size: int = 1, 
        *,
        idxs: ListLike = None,
        rate: float = None,
        **kwargs,
    ) -> dict[str, Union[ObsBatch, ActBatch, th.FloatTensor]]:
        """随机采样

        Args
        ----------
        batch_size : int, optional
            样本容量, 默认1.
        
        KwArgs
        ----------
        idxs : ListLike, optional
            若传入样本索引, 则按索引采样(此时batch_size不起作用), 否则根据样本容量随机生成索引, 默认None.
        rate : float, optional
            用于PER更新参数 beta, 默认None.
            rate = learn_steps / max_train_steps
            beta = beta0 + (1-beta0) * rate

        Returns
        -------
        Dict[str, Union[ObsBatch, ActBatch, th.FloatTensor]]
            要求返回key为 "s", "a", "r", "s_", "done", "IS_weight", ... 的GPU版Tensor/MixedTensor存储形式
        """  
        raise NotImplementedError

    def __getitem__(self, index):
        """索引样本\n
           即 batch = buffer[index] 与 batch = buffer.sample(idxs=index) 效果相同
        """
        if isinstance(index, int): index = [index]
        return self.sample(idxs=index)
    
    # 3.PER功能
    def update_priorities(self, td_errors: np.ndarray):
        """使用TD误差更新PER优先级"""
        pass
    
    @property
    def is_per(self) -> bool:
        """是否是PER缓存"""
        return False
    
    # 4.一个Obs转换成样本
    @abstractmethod
    def state_to_tensor(self, state: Obs, use_rnn=False) -> ObsBatch:
        """算法的select_action和export接口调用, 用于将1个state转换成batch_size=1的张量
        use_rnn = False : (*state_shape, ) -> (1, *state_shape)
        use_rnn = True : (*state_shape, ) -> (1, 1, *state_shape)
        """
        raise NotImplementedError
        # TODO 若想支持混合动作空间, 需定义 action_to_numpy 方法
    
    # 5.IO接口
    def save(self, data_dir: PathLike, buffer_id: Union[int, str] = None):
        """存储buffer\n
        存储在 data_dir / buffer_id 或 data_dir 中
        """
        pass

    def load(self, data_dir: PathLike, buffer_id: Union[int, str] = None):
        """读取buffer\n
        存储在 data_dir / buffer_id 或 data_dir 中
        """
        pass

    # 6.PyTorch功能
    def to(self, device: DeviceLike):
        """返回的样本张量设置到device上"""
        self.device = device
        return self

    def cuda(self, cuda_id=None):
        """返回的样本设置为cuda张量"""
        device = 'cpu' if not th.cuda.is_available() else 'cuda' if cuda_id is None else 'cuda:' + str(cuda_id)
        self.to(device)
        return self

    def cpu(self):
        """返回的样本设置为cpu张量"""
        self.to('cpu')
        return self
    
    
    


#----------------------------- ↓↓↓↓↓ Soft Actor-Critic ↓↓↓↓↓ ------------------------------#

# Q网络
class SAC_Critic(nn.Module):
    def __init__(self, encoder: nn.Module, q1_layer: nn.Module, q2_layer: nn.Module):
        """设置SAC的Critic\n
        要求encoder输入为obs, 输出为 (batch, dim) 的特征 x.\n
        要求q1_layer和q2_layer输入为 (batch, dim + act_dim) 的拼接向量 cat[x, a], 输出为 (batch, 1) 的 Q.\n
        """
        super().__init__()
        self.encoder_layer = deepcopy(encoder)
        self.q1_layer = deepcopy(q1_layer)
        self.q2_layer = deepcopy(q2_layer)

    def forward(self, obs, act):
        feature = self.encoder_layer(obs) # (batch, feature_dim)
        x = th.cat([feature, act], -1)
        Q1 = self.q1_layer(x)
        Q2 = self.q2_layer(x)
        return Q1, Q2



# PI网络
class SAC_Actor(nn.Module):
    def __init__(self, encoder: nn.Module, mu_layer: nn.Module, log_std_layer: nn.Module, log_std_max=2.0, log_std_min=-20.0):
        """设置SAC的Actor\n
        要求encoder输入为obs, 输出为 (batch, dim) 的特征 x.\n
        要求log_std_layer和mu_layer输入为 x, 输出为 (batch, act_dim) 的对数标准差和均值.\n
        """
        super().__init__()
        self.encoder_layer = deepcopy(encoder)
        self.mu_layer = deepcopy(mu_layer)
        self.log_std_layer = deepcopy(log_std_layer)
        self.LOG_STD_MAX = log_std_max
        self.LOG_STD_MIN = log_std_min

    def forward(self, obs, deterministic=False, with_logprob=True):
        feature = self.encoder_layer(obs) # (batch, feature_dim)
        mu = self.mu_layer(feature)
        log_std = self.log_std_layer(feature)
        log_std = th.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = th.exp(log_std)
        # 策略分布
        dist = Normal(mu, std)
        if deterministic: u = mu
        else: u = dist.rsample()
        a = th.tanh(u)
        # 计算动作概率的对数
        if with_logprob:
            # 1.SAC论文通过u的对数概率计算a的对数概率公式:
            "logp_pi_a = (dist.log_prob(u) - th.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)"
            # 2.SAC原文公式有a=tanh(u), 导致梯度消失, 将tanh公式展开:
            logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True) # (batch, 1)
        else:
            logp_pi_a = None
        return a, logp_pi_a # (batch, act_dim) and (batch, 1)

    def act(self, obs, deterministic=False) -> np.ndarray[any, float]: # NOTE 不支持混合动作空间
        self.eval()
        with th.no_grad():
            a, _ = self.forward(obs, deterministic, False)
        self.train()
        return a.cpu().numpy().flatten() # (act_dim, ) ndarray
    


# SAC-Auto算法
class SAC_Agent:
    """Soft Actor-Critic (arXiv: 1812) 算法"""
   
    def __init__(
        self, 
        env: GymEnv,                # gym环境 或 cfg参数
        *,
        gamma: float = 0.99,        # 折扣因子 γ
        alpha: float = 0.2,         # 温度系数 α
        batch_size: int = 128,      # 样本容量
        update_after: int = 1000,   # 训练开始，batch_size <= update_after <= memory_size

        lr_decay_period: int = None, # 学习率衰减周期, None不衰减
        lr_critic: float = 1e-3,     # Q 学习率
        lr_actor: float = 1e-3,      # π 学习率
        tau: float = 0.005,          # target Q 软更新系数 τ
        q_loss_cls = nn.MSELoss,     # Q 损失函数类型(use_per=True时该设置无效)
        grad_clip: float = None,     # Q网络梯度裁剪范围, None不裁剪

        adaptive_alpha: bool = True,     # 是否自适应温度系数
        target_entropy: float = None,    # 自适应温度系数目标熵, 默认: -dim(A)
        lr_alpha: float = 1e-3,          # α 学习率
        alpha_optim_cls = th.optim.Adam, # α 优化器类型

        device: DeviceLike = th.device("cuda" if th.cuda.is_available() else "cpu"), # 计算设备
    ): 
        """
        Args:
            env (GymEnv): Gym环境实例, 或包含observation_space和action_space的数据类.
        KwArgs:
            gamma (float): 累积奖励折扣率. 默认0.99.
            alpha (float): 初始温度系数. 默认0.2.
            batch_size (int): 样本容量. 默认128.
            update_after (int): 训练开始步数. 默认1000.
            lr_decay_period (int): 学习率衰减到原来的0.1倍的周期. 默认None不衰减.
            lr_critic (float): Q函数学习率. 默认0.001.
            lr_actor (float): Pi函数学习率. 默认0.001.
            tau (float): 目标Q函数软更新系数. 默认0.005.
            q_loss_cls (TorchLossClass): Q函数的损失函数. 默认MSELoss.
            grad_clip (float): Q函数梯度裁剪范围. 默认None不裁剪.
            adaptive_alpha (bool): 是否自适应温度系数. 默认True.
            target_entropy (float): 目标策略熵. 默认-dim(A).
            lr_alpha (float): 温度系数学习率. 默认0.001.
            alpha_optim_cls (TorchOptimizerClass): 温度系数优化器. 默认Adam.
            device (DeviceLike): 训练设备. 默认cuda0.
        """
        assert isinstance(env.action_space, GymBox), "SAC-Auto算法的动作空间只能是Box"
        self.device = device
        # 环境参数
        self.obs_space = env.observation_space
        self.act_space = env.action_space
        self.num_actions = np.prod(self.act_space.shape)
        # SAC参数初始化
        self.gamma = gamma
        self.batch_size = int(batch_size)
        self.update_after = int(update_after)
        # DL参数初始化
        self.lr_decay_period = lr_decay_period
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.tau = tau
        # ReplayBuffer初始化
        self.__set_buffer = False
        self.buffer = BaseBuffer()
        # 神经网络初始化
        self.__set_nn = False
        self.actor = None
        self.q_critic = None
        self.target_q_critic = None
        self.actor_optimizer = None
        self.q_critic_optimizer = None
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
            self.alpha_optimizer = alpha_optim_cls([self.log_alpha], lr = lr_alpha)
            self.lr_alpha = lr_alpha
        # 其它参数
        self.learn_counter = 0
    
    # 0.Torch接口
    def to(self, device: DeviceLike):
        """算法device设置"""
        assert self.__set_nn, "未设置神经网络!"
        assert self.__set_buffer, "未设置ReplayBuffer!"
        self.device = device
        self.buffer.to(device)
        self.actor.to(device)
        self.q_critic.to(device)
        self.target_q_critic.to(device)
        if self.adaptive_alpha:
            self.log_alpha.to(device)
            self.target_entropy.to(device)
        return self

    def cuda(self, cuda_id=None):
        """算法device转换到cuda上"""
        device = 'cpu' if not th.cuda.is_available() else 'cuda' if cuda_id is None else 'cuda:' + str(cuda_id)
        self.to(device)
        return self

    def cpu(self):
        """算法device转换到cpu上"""
        self.to('cpu')
        return self
    
    # 1.IO接口
    def save(self, data_dir: PathLike):
        """存储算法"""
        assert self.__set_nn, "未设置神经网络!"
        assert self.__set_buffer, "未设置ReplayBuffer!"
        data_dir = Path(data_dir)
        model_dir = data_dir/'state_dict'
        model_dir.mkdir(parents=True, exist_ok=True)
        th.save(self.actor.state_dict(), model_dir/'actor.pth')
        th.save(self.q_critic.state_dict(), model_dir/'critic.pth')
        th.save(self.target_q_critic.state_dict(), model_dir/'target_critic.pth')
        self.buffer.save(data_dir/'buffer')
        #存储 温度系数/优化器/算法参数等, 代码略
        
    def load(self, data_dir: PathLike):
        """加载算法"""
        assert self.__set_nn, "未设置神经网络!"
        assert self.__set_buffer, "未设置ReplayBuffer!"
        data_dir = Path(data_dir)
        self.actor.load_state_dict(th.load(data_dir/'state_dict'/'actor.pth', map_location=self.device))
        self.q_critic.load_state_dict(th.load(data_dir/'state_dict'/'critic.pth', map_location=self.device))
        self.target_q_critic.load_state_dict(th.load(data_dir/'state_dict'/'target_critic.pth', map_location=self.device))
        self.buffer.load(data_dir/'buffer')
        #加载 温度系数/优化器/算法参数等, 代码略

    def export(
        self, 
        file: PathLike, 
        map_device: DeviceLike = 'cpu', 
        use_stochastic_policy: bool = True, 
        output_logprob: bool = False
    ):
        """导出onnx策略模型 (可通过 https://netron.app 查看模型计算图)\n
        Args:
            file (PathLike): 模型文件名.
            map_device (DeviceLike): 模型计算设备. 默认'cpu'.
            use_stochastic_policy (bool): 是否使用随机策略模型. 默认True.
            output_logprob (bool): 模型是否计算SAC的策略信息熵. 默认False.
        """
        assert self.__set_nn, "未设置神经网络!"
        file = Path(file).with_suffix('.onnx')
        file.parents[0].mkdir(parents=True, exist_ok=True)
        # 输入输出设置
        device = deepcopy(self.device)
        self.to(map_device)
        obs_tensor = self.state_to_tensor(self.obs_space.sample())
        dummy_input = (obs_tensor, not use_stochastic_policy, output_logprob) # BUG use_rnn需添加 h 或 h+C
        input_names, output_names = self._get_onnx_input_output_names(False) # BUG use_rnn未实现
        dynamic_axes, axes_name = self._get_onnx_dynamic_axes(input_names, output_names)
        if output_logprob:
            output_names += ['logprob']
            dynamic_axes['logprob'] = axes_name
        # 模型部署
        self.actor.eval()
        th.onnx.export(
            self.actor, 
            dummy_input,
            file, 
            input_names = input_names,
            output_names = output_names,
            dynamic_axes = dynamic_axes,
            export_params = True,
            verbose = False,
            opset_version = 11, # 11版之后才支持Normal运算
        )
        self.actor.train()
        self.to(device)

    def _get_onnx_input_output_names(self, use_rnn=False):
        """获取onnx的输入输出名"""
        if isinstance(self.obs_space, GymDict):
            input_names = [str(k) for k in self.obs_space]
        elif isinstance(self.obs_space, GymTuple):
            input_names = ['observation'+str(i) for i in range(len(self.obs_space))]
        else:
            input_names = ['observation']
        output_names = ['action'] # NOTE 暂不支持混合动作空间
        if use_rnn:
            input_names += ['old_hidden'] # BUG 不支持 LSTM, 需添加 cell_state
            output_names += ['new_hidden']
        return input_names, output_names

    def _get_onnx_dynamic_axes(self, onnx_input_names, onnx_output_names):
        """获取onnx的动态轴"""
        # data shape: (batch_size, seq_len, *shape)
        # hidden shape: (num_directions*num_layer, batch_size, hidden_size)
        if 'old_hidden' in onnx_input_names:
            data_axes_name = {0: 'batch_size', 1: 'seq_len'}
        else:
            data_axes_name = {0: 'batch_size'}
        axes_dict = {}
        for k in onnx_input_names+onnx_output_names:
            if k in {'old_hidden', 'new_hidden'}:
                axes_dict[k] = {1: 'batch_size'}
            else:
                axes_dict[k] = data_axes_name
        return axes_dict, data_axes_name

    # 2.神经网络设置接口
    def set_nn(
        self, 
        actor: SAC_Actor, 
        critic: SAC_Critic, 
        *, 
        actor_optim_cls = th.optim.Adam, 
        critic_optim_cls = th.optim.Adam, 
        copy: bool = True
    ):
        """设置神经网络, 要求为SAC_Actor/SAC_Critic的实例对象, 或结构相同的鸭子类的实例对象"""
        self.__set_nn = True
        self.actor = deepcopy(actor) if copy else actor
        self.actor.train().to(self.device)
        self.q_critic = deepcopy(critic) if copy else critic
        self.q_critic.train().to(self.device) # Twin Q Critic
        self.target_q_critic = self._build_target(self.q_critic)
        self.actor_optimizer = actor_optim_cls(self.actor.parameters(), self.lr_actor)
        self.q_critic_optimizer = critic_optim_cls(self.q_critic.parameters(), self.lr_critic)

    # 3.经验回放设置接口
    def set_buffer(self, buffer: BaseBuffer):
        """设置经验回放, 要求为BaseBuffer的派生类的实例对象, 或结构相同的鸭子类的实例对象"""
        self.__set_buffer = True
        self.buffer = buffer
    
    def store_memory(
        self, 
        transition: tuple[Obs, Act, float, Obs, bool], 
        terminal: bool = None, 
        **kwargs
    ):
        """经验存储\n
        Args:
            transition (tuple): (s, a, r, s_, done)元组, 顺序不能变.
            terminal (bool): DRQN/R2D2等RNN算法控制参数, 控制Buffer时间维度指针跳转.
            **kwargs: Buffer.push的其它控制参数.
            注意: done表示成功/失败/死亡等, 此时没有下一个状态s_; terminal表示回合结束(新gym的truncated参数), 可能是超时/越界等导致的, 此时有下一个状态s_.
        """
        assert self.__set_buffer, "未设置ReplayBuffer!"
        self.buffer.push(transition, terminal, **kwargs)
 
    def replay_memory(self, batch_size: int, **kwargs):
        """经验回放\n
        Args:
            batch_size (int): 样本容量.
            **kwargs: Buffer.sample的控制参数, 如优先经验回放需要传入rate = learn_step/total_step 更新Buffer的alpha/beta参数.
        Returns:
            batch = {'s': ObsBatch, 'a': ActBatch, 'r': FloatTensor, 's_': ObsBatch, 'done': FloatTensor, ...}\n
            若为PER字典的key还要包含'IS_weight'.
        """
        return self.buffer.sample(batch_size, **kwargs)
    
    @property
    def buffer_len(self) -> int:
        """当前存储的容量"""
        return len(self.buffer)
    
    @property
    def use_per(self) -> bool:
        """是否优先经验回放"""
        return self.buffer.is_per
        
    # 4.决策模块接口
    def state_to_tensor(self, state: Obs) -> ObsBatch:
        """状态升维并转换成Tensor"""
        return self.buffer.state_to_tensor(state, use_rnn=False) # (1, *state_shape) tensor GPU
    
    def select_action(self, state: Obs, *, deterministic=False, **kwargs) -> np.ndarray:
        """选择动作 -> [-1, 1]"""
        assert self.__set_nn, "未设置神经网络!"
        state = self.state_to_tensor(state)
        return self.actor.act(state, deterministic) # (act_dim, ) ndarray
    
    def random_action(self) -> np.ndarray:
        """随机动作 -> [-1, 1]"""
        action = self.act_space.sample()
        lb, ub = self.act_space.low, self.act_space.high
        action = 2 * (action - lb) / (ub - lb) - 1
        return np.clip(action, -1.0, 1.0) # (act_dim, ) ndarray

    # 5.强化学习接口
    def learn(self, **kwargs) -> dict[str, Union[float, None]]:
        """Soft Actor-Critic\n
        1.优化Critic
            min J(Q) = LOSS[ Q(s, a) - Q* ]\n
            Q* = r + (1-d) * γ * V(s_, a*)\n
            V(s_, a*) = Qt(s_, a*) - α*log π(a*|s_)\n
        2.优化Actor
            min J(π) = -V(s, a^)\n
            V(s, a^) = α*log π(a^|s) - Q(s, a^)\n
        3.优化Alpha
            min J(α) = -α * (log π(a^|s) + H0)\n
            min J(α) = -logα * (log π(a^|s) + H0) -> 速度更快\n
        """
        assert self.__set_nn, "未设置神经网络!"
        if self.buffer_len < self.batch_size or self.buffer_len < self.update_after:    
            return {'q_loss': None, 'actor_loss': None, 'alpha_loss': None, 'q': None, 'alpha': None}
        self.learn_counter += 1
        
        ''' experience repaly '''
        batch = self.replay_memory(self.batch_size, **kwargs) # return tensor GPU

        ''' critic update '''
        #* J(Q) = E_{s_t~D, a_t~D, s_t+1~D, a_t+1~π_t+1}[0.5*[ Q(s_t, a_t) - [r + (1-d)*γ* [ Q_tag(s_t+1,a_t+1) - α*logπ_t+1 ] ]^2 ]
        q_loss, Q_curr = self._compute_qloss(batch)
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.q_critic.parameters(), self.grad_clip)
        self.q_critic_optimizer.step()

        ''' actor update '''
        #* J(π) = E_{s_t~D, a~π_t}[ α*logπ_t(a|π_t) - Q(s_t, a) ] 
        self._freeze_network(self.q_critic)
        a_loss, log_pi = self._compute_ploss(batch)
        self._optim_step(self.actor_optimizer, a_loss)
        self._unfreeze_network(self.q_critic)

        ''' alpha update '''
        #* J(α) = E_{a~π_t}[ -α * ( logπ_t(a|π_t) + H0 ) ]
        if self.adaptive_alpha:
            alpha_loss = -(self.log_alpha * (log_pi.detach() + self.target_entropy)).mean()        # 收敛较快, 计算较快
            #alpha_loss = -(self.log_alpha.exp() * (log_pi.detach() + self.target_entropy)).mean() # 收敛用的episode较大, 且计算速度慢
            self._optim_step(self.alpha_optimizer, alpha_loss)
            self.alpha = self.log_alpha.exp().item()
            alpha_loss = alpha_loss.item() # logging
        else:
            alpha_loss = None

        ''' target update '''
        self._soft_update(self.target_q_critic, self.q_critic, self.tau)
        
        ''' lr decay '''
        if self.lr_decay_period:
            self._lr_decay(self.actor_optimizer)
            self._lr_decay(self.q_critic_optimizer)
            if self.adaptive_alpha:
                self._lr_decay(self.alpha_optimizer)
        
        return {'q_loss': q_loss.item(), 'actor_loss': a_loss.item(), 'alpha_loss': alpha_loss, 
                'q': Q_curr.mean().item(), 'alpha': self.alpha}
    
    # 6.SAC损失函数
    def _compute_qloss(self, batch) -> tuple[th.Tensor, th.Tensor]:
        """计算Q-Critic(连续)或Q-Net(离散)的损失, 返回Loss和当前Q值"""
        s, a, r, s_, done = batch["s"], batch["a"], batch["r"], batch["s_"], batch["done"]
        #* SAC: Q_targ = E[ r + (1-d) * γ * V_next ]
        #* SAC: V_next = E[ Q_next - α*logπ_next ]
        with th.no_grad():
            a_, log_pi_next = self.actor(s_)                # (m, act_dim), (m, 1) GPU no grad
            Q1_next, Q2_next = self.target_q_critic(s_, a_) # (m, 1)
            Q_next = th.min(Q1_next, Q2_next)               
            Q_targ = r + (1.0 - done) * self.gamma * (Q_next - self.alpha*log_pi_next) # (m, 1)
        Q1_curr, Q2_curr = self.q_critic(s, a)              # (m, 1) GPU with grad

        if self.use_per:
            IS_weight = batch["IS_weight"]
            td_err1, td_err2 = Q1_curr-Q_targ, Q2_curr-Q_targ
            q_loss = (IS_weight * (td_err1 ** 2)).mean() + (IS_weight * (td_err2 ** 2)).mean() # () 注意: mean一定加在最外面！！！！
            self.buffer.update_priorities(td_err1.detach().cpu().numpy().flatten()) # 更新优先级 td err: (m, ) ndarray
        else:
            q_loss = self.q_loss(Q1_curr, Q_targ) + self.q_loss(Q2_curr, Q_targ) # ()

        return q_loss, Q1_curr

    def _compute_ploss(self, batch) -> tuple[th.Tensor, th.Tensor]:
        """计算Actor的损失和logπ, 返回Loss和logπ"""
        state = batch["s"]
        new_action, log_pi = self.actor(state)    # (m, act_dim), (m, 1) GPU with grad
        Q1, Q2 = self.q_critic(state, new_action) # (m, 1) GPU freeze grad 
        Q = th.min(Q1, Q2)                        # (m, 1) GPU freeze grad
        a_loss = (self.alpha*log_pi - Q).mean()
        return a_loss, log_pi

    # 7.功能函数
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

