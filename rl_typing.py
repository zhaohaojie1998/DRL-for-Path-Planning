# -*- coding: utf-8 -*-
"""
RL类型提示
 Created on Sat Nov 04 2023 15:37:28
 Modified on 2023-11-4 15:37:28
 
 @auther: HJ https://github.com/zhaohaojie1998
"""
import numpy as np
import torch as th
from gym import spaces
from typing import Union, Literal, Optional


__all__ = [
    #官方类型
    "Union",
    "Optional",
    "Literal",

    #类型声明
    "ListLike",
    "PathLike",
    "DeviceLike",
    "TorchLoss",
    "TorchOptimizer",
    "GymEnv",
    "GymBox",
    "GymDiscrete",
    "GymTuple",
    "GymDict",
    
    #输入输出声明
    "ObsSpace",
    "ActSpace",
    "Obs",
    "Act",
    "ObsBatch",
    "ActBatch",
]

#----------------------------- ↓↓↓↓↓ 类型声明 ↓↓↓↓↓ ------------------------------#
from os import PathLike
ListLike = Union[list, np.ndarray]

DeviceLike = Union[th.device, str, None]
from torch.nn.modules.loss import _Loss as TorchLoss
from torch.optim import Optimizer as TorchOptimizer

from gym import Env as GymEnv
from gym.spaces import Box as GymBox
from gym.spaces import Discrete as GymDiscrete
from gym.spaces import Tuple as GymTuple
from gym.spaces import Dict as GymDict

ObsSpace = spaces.Space                                             # 状态/观测空间: 任意
ActSpace = Union[spaces.Box, spaces.Discrete, spaces.MultiDiscrete] # 动作/控制空间: Box连续, Discrete编码, MultiDiscrete离散

_MetaObs = Union[int, np.ndarray]
_MixedObs = Union[dict[any, _MetaObs], tuple[_MetaObs, ...], list[_MetaObs]]
Obs = Union[_MetaObs, _MixedObs] # 状态/观测: int, array, 混合
Act = Union[int, np.ndarray]     # 动作/控制: int为编码控制量(DiscreteAct), array为连续(BoxAct)/离散(MultiDiscreteAct)控制量

_MetaObsBatch = th.FloatTensor
_MixedObsBatch = Union[dict[any, _MetaObsBatch], tuple[_MetaObsBatch, ...], list[_MetaObsBatch]]
ObsBatch = Union[_MetaObsBatch, _MixedObsBatch] # 神经网络输入: FloatTensor或其混合形式
ActBatch = Union[th.FloatTensor, th.LongTensor] # 神经网络输出: FloatTensor为连续控制量, LongTensor为编码/离散控制量
