import torch
import numpy as np
import torch.nn as nn
import torchvision 
from torchvision import transforms, datasets
import math
from copy import deepcopy

def quantisize_tensor(x, num_bits = 8):
        #量化范围
        qmin = 0.
        qmax = 2.**num_bits - 1.
        #计算最大值和最小值
        val_min, val_max = x.min(), x.max()
        
        #计算缩放系数，量化区间的压缩比例
        scale = (val_max - val_min) / (qmax - qmin)
        #uint8为不一0为对称点，因此需要重新确定对称点
        zero_point = qmax - val_max/scale
        #
        q_x = zero_point + x / scale
        #数据截断
        q_x.clamp_(qmin, qmax).round_()
        q_x = q_x.round().byte()
        return q_x, scale, zero_point

def dequant(q_x, scale, zero_point):
        return scale * (q_x.float() - zero_point)
        
w = torch.randn(3,4)
print(w)
# print(*quantize_tensor(w))
print(dequant(*quantisize_tensor(w)))
