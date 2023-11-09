import random
import numpy as np
import torch
from timm.models.layers import to_2tuple


class RandomMaskingGenerator:  # 生成一个mask而已
    def __init__(self, input_channel, len_mask, mask_ratio):
        """
        :param input_channel:bands,dim=1
        :param len_mask: len of mask
        :param mask_ratio: MAE default 0.75
        """
        self.input_channel = input_channel
        L = len(input_channel)  # 输入的通道总数
        parity = L % len_mask  # 判断能否被全部mask
        if parity is 0:
            print("通道数可被整除，通道数为：{}".format(L))
        else:
            raise ValueError("通道数不能被整除")
        self.num_patches = L * mask_ratio  # 数量

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            len(self.input_channel), int(self.num_patches)
        )
        return repr_str

    def __call__(self, *args, **kwargs):
        mask = np.hstack([
            np.zeros((1, len(self.input_channel)-int(self.num_patches))),
            np.ones((1, int(self.num_patches)))
        ])
        np.random.shuffle(mask)
        return mask


