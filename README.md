# PyTorch版SAC-Auto强化学习模块

## 零.SAC-Auto算法:

###### 自定义程度高的SAC-Auto算法，支持部署策略模型、备份训练过程、多源观测融合、PER等功能

论文：《Soft Actor-Critic Algorithms and Applications （arXiv: 1812) 》# 不是1801版

| 算法构成     | 说明                 |
| ------------ | -------------------- |
| rl_typing.py | 强化学习数据类型声明 |
| sac_agent.py | SAC-Auto算法         |

### (0).SAC_Agent模块

###### SAC-Auto算法主模块

##### 0.初始化接口

```python
agent = SAC_Agent(env, kwargs=...)      # 初始化算法, 并设置SAC的训练参数
agent.set_buffer(buffer)                # 为算法自定义replay buffer
agent.set_nn(actor, critic, kwargs=...) # 为算法自定义神经网络
# 更多具体接口信息通过help函数查看DocString
```

##### 1.Torch接口

```python
agent.to('cpu') # 将算法转移到指定设备上
agent.cuda(0)   # 将算法转移到cuda0上运算
agent.cpu()     # 将算法转移到cpu上运算
```

##### 2.IO接口

```python
agent.save('./训练备份')              # 存储算法训练过程checkpoint
agent.load('./训练备份')              # 加载算法训练过程checkpoint
agent.export('策略.onnx', kwargs=...) # 部署训练好的onnx策略模型
```

##### 3.训练交互接口

```python
act_array = agent.select_action(obs, kwargs=...) # 环境交互, 基于策略选择-1~1的随机/确定动作
act_array = agent.random_action()                # 环境随机探索, 完全随机产生-1~1的动作
agent.store_memory(transition, kwargs=...)       # 存储环境转移元组(s, a, r, s_, done)
info_dict = agent.learn(kwargs=...)              # 进行一次SAC优化, 返回Loss/Q函数/...
```

##### 4.其余接口/属性 (非用户调用接口，可在派生SAC_Agent模块中覆写)

```python
obs_tensor = agent.state_to_tensor(obs, kwargs=...) # 将Gym返回的1个obs转换成batch_obs, 用于处理混合输入情况, 默认跟随buffer设置
batch_dict = agent.replay_memory(batch_size, kwargs=...) # 经验回放, 用于实现花样经验回放, 默认跟随buffer设置
agent.buffer_len # 算法属性, 查看当前经验个数, 默认跟随buffer设置
agent.use_per # 算法属性, 查看是否使用PER, 默认跟随buffer设置
```

### (1).SAC_Actor模块和SAC_Critic模块

###### 实现自定义 **观测Encoder** + **策略函数** + **Q函数**

##### 0.自定义神经网络要求

- 要求 **观测Encoder** 输入为观测 *batch_obs* 张量，输出形状为(batch, feature_dim)的特征 *batch_feature* 张量。要求forward函数只接受一个位置参数obs，混合观测要求传入的obs为张量字典dict[any, Tensor] / 张量列表list[Tensor] / 张量元组tuple[Tensor, ...]。
- 要求 **策略函数** 输入为特征 *batch_feature* 张量，输出形状为(batch, action_dim)的未经tanh激活的均值 *batch_mu* 张量和对数标准差 *batch_logstd* 张量。要求forward函数只接受一个位置参数feature，形状为(batch, feature_dim)。
- 要求 **Q函数** 输入为特征 *batch_feature* 张量+动作 *batch_action* 张量，输出形状为(batch, 1)的Q值 *batch_q* 张量。要求forward函数只接受一个位置参数 *feature_and_action*，形状为(batch, feature_dim+action_dim)。

##### 1.自定义神经网络示例

```python
encoder_net = MyEncoder()                   # 自定义观测编码器
mu_net, logstd_net = MyPolicy(), MyPolicy() # 自定义策略函数
q1_net, q2_net = MyQfun(), MyQfun()         # 自定义双Q函数
actor = SAC_Actor(encoder_net, mu_net, logstd_net, kwargs=...) # 设置自定义actor网络
critic = SAC_Critic(encoder_net, q1_net, q2_net)               # 设置自定义critic网络
agent.set_nn(
    actor, 
    critic, 
    actor_optim_cls = th.optim.Adam, 
    critic_optim_cls = th.optim.Adam, 
    copy = True
) # 为算法设置神经网络
```

### (2).BaseBuffer模块

实现自定义经验回放，可自定义存储不同数据类型的混合观测数据（进行一些多传感器数据融合的端到端控制问题求解），也可自定义实现PER等功能。

要求在派生类中实现以下抽象方法（输入参数和返回数据的格式参考DocString)，可参考demo_train.py中派生类实现方法：

| **必须实现的方法**        | **功能**                                                     |
| ------------------------- | ------------------------------------------------------------ |
| reset                     | 重置经验池（Off-Policy算法一般用不到），也可用于初始化经验池（生成转移元组collections） |
| push                      | 经验存储：存入环境转移元组 *(s, a, r, s_, done)* ，其中状态 *s* 和下一个状态 *s_* （或观测 *obs* ）为array（或混合形式dict[any, array]、list[array]、tuple[array, ...]），动作 *a* 为array，奖励 *r* 为float， *s_* 是否存在 *done* 为bool。 |
| sample                    | 经验采样：要求返回包含关键字 *'s','a','r','s_','done'* 的 *batch* 字典， *batch* 的每个key对应value为Tensor（或dict[any, Tensor]、list[Tensor]、tuple[Tensor, ...]）；PER的batch还要包含关键字 *'IS_weight'* ，对应的value为Tensor。 |
| state_to_tensor           | 数据升维并转换：将Gym输出的1个 *obs* 转换成 *batch obs* ，要求返回Tensor（或混合形式dict[any, Tensor]、list[Tensor]、tuple[Tensor, ...]）。 |
| **非必须实现的方法/属性** | **功能**                                                     |
| save                      | 存储buffer数据，用于保存训练进度，可省略                     |
| load                      | 加载buffer数据，用于加载训练进度，可省略                     |
| update_priorities         | 用于更新PER的优先级，非PER可省略                             |
| is_per（属性）            | 是否是PER回放，默认False                                     |
| is_rnn（属性）            | 是否RNN按episode回放，默认False                              |
| nbytes（属性）            | 用于查看经验池占用内存，默认0                                |

## 一.路径规划环境SAC应用示例:

###### 路径规划环境包 path_plan_env

| 包含的模块               | 说明                                                 |
| ------------------------ | ---------------------------------------------------- |
| LidarModel               | 激光雷达模拟（基于东北天坐标系）                     |
| NormalizedActionsWrapper | 环境装饰器：非-1~1动作空间归一化，用于与算法适配     |
| DynamicPathPlanning      | 动力学路径规划环境（动作空间-1~1，基于东天南坐标系） |
| StaticPathPlanning       | 路径搜索环境（动作空间非-1~1）                       |

### (0).环境接口

###### gym标准接口格式，初始化时可指定使用老版gym接口风格或新版gym接口风格

```python
# 实例化环境
from path_plan_env import DynamicPathPlanning
env = DynamicPathPlanning(kwargs=...)
# 训练/测试交互
obs, info = env.reset(kwargs=...) # new gym style
obs = env.reset(kwargs=...)       # old gym style
while 1:
    try:
        env.render(kwargs=...) # 可视化路径规划(测试)
        act = np.array([...]) # shape=(act_dim, ) range∈-1~1
        obs, rew, done, truncated, info = env.step(act, kwargs=...) # new gym style
        obs, rew, done, info = env.step(act, kwargs=...)            # old gym style
    except AssertionError:
        env.plot("fig.png", kwargs=...) # 输出规划结果(训练)
        break
```

### (1).路径搜索环境（StaticPathPlanning）

###### 几何层面规划，直接找几个点组成路径，学习组成路径的点

<img src="图片/Result.png" style="zoom:80%;" />

### (2).动力学路径规划环境（DynamicPathPlanning）

###### 动力学层面规划，学习控制量

##### 0.雷达感知模型

<img src="图片/Lidar.gif" style="zoom:200%;" />

##### 1.训练结果

<img src="图片/amagi1.png" alt="img" style="zoom: 67%;" />

<img src="图片/amagi2.png" alt="img" style="zoom: 80%;" />

##### 2.仿真结果



## 二.**Requirement**:

python >= 3.9

SAC算法依赖项：

gym >= 0.21.0 （数据结构API）

numpy >= 1.22.3 （数组运算API）

pytorch >= 1.10.2 （深度学习API）

onnx >= 1.13.1 （模型部署API）

onnxruntime >= 1.15.1 （模型推理API）

非SAC算法依赖项：

tensorboard （训练日志记录）

scipy >= 1.7.3 （自定义Env数值积分）

shapely >= 2.0.1 （自定义Env障碍表示）

matplotlib >= 3.5.1 （自定义Env可视化）

###### 广告：

[Path-Planning: 路径规划算法，A*、Dijstra、Hybrid A*等经典路径规划](https://github.com/zhaohaojie1998/A-Star-for-Path-Planning)

<img src="图片/ad1.png" style="zoom: 67%;" />

[Grey-Wolf-Optimizer-for-Path-Planning: 灰狼优化算法路径规划、多智能体/多无人机航迹规划](https://github.com/zhaohaojie1998/Grey-Wolf-Optimizer-for-Path-Planning)

<img src="图片/ad2.png" style="zoom: 50%;" />
