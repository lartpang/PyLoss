# -*- coding: utf-8 -*-
# @Time    : 2020/12/13
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import time

import torch
from torch.nn.functional import one_hot


def bhw_to_onehot_by_for(bhw_tensor: torch.Tensor, num_classes: int):
    """
    Args:
        bhw_tensor: b,h,w
        num_classes:

    Returns: b,h,w,num_classes
    """
    assert bhw_tensor.ndim == 3, bhw_tensor.shape
    assert num_classes > bhw_tensor.max(), torch.unique(bhw_tensor)
    one_hot = bhw_tensor.new_zeros(size=(num_classes, *bhw_tensor.shape))
    for i in range(num_classes):
        one_hot[i, bhw_tensor == i] = 1
    one_hot = one_hot.permute(1, 2, 3, 0)
    return one_hot


def bhw_to_onehot_by_for_V1(bhw_tensor: torch.Tensor, num_classes: int):
    """
    Args:
        bhw_tensor: b,h,w
        num_classes:

    Returns: b,h,w,num_classes
    """
    assert bhw_tensor.ndim == 3, bhw_tensor.shape
    assert num_classes > bhw_tensor.max(), torch.unique(bhw_tensor)
    one_hot = bhw_tensor.new_zeros(size=(*bhw_tensor.shape, num_classes))
    for i in range(num_classes):
        one_hot[..., i][bhw_tensor == i] = 1
    return one_hot


def bhw_to_onehot_by_scatter(bhw_tensor: torch.Tensor, num_classes: int):
    """
    Args:
        bhw_tensor: b,h,w
        num_classes:

    Returns: b,h,w,num_classes
    """
    assert bhw_tensor.ndim == 3, bhw_tensor.shape
    assert num_classes > bhw_tensor.max(), torch.unique(bhw_tensor)
    batch_size, h, w = bhw_tensor.shape
    # bhw,c
    one_hot = torch.zeros(size=(batch_size * h * w, num_classes)).scatter_(
        dim=1, index=bhw_tensor.reshape(-1, 1), value=1
    )
    one_hot = one_hot.reshape(batch_size, h, w, num_classes)
    return one_hot


def bhw_to_onehot_by_scatter_V1(bhw_tensor: torch.Tensor, num_classes: int):
    """
    Args:
        bhw_tensor: b,h,w
        num_classes:

    Returns: b,h,w,num_classes
    """
    assert bhw_tensor.ndim == 3, bhw_tensor.shape
    assert num_classes > bhw_tensor.max(), torch.unique(bhw_tensor)
    # self[i][j][k][index[i][j][k][l]] = value  # 实际上就是便利了index的所有元素，用其索引调整self
    one_hot = torch.zeros(size=(*bhw_tensor.shape, num_classes)).scatter_(
        dim=-1, index=bhw_tensor[..., None], value=1
    )
    return one_hot


def bhw_to_onehot_by_index_select(bhw_tensor: torch.Tensor, num_classes: int):
    """
    Args:
        bhw_tensor: b,h,w
        num_classes:

    Returns: b,h,w,num_classes
    """
    assert bhw_tensor.ndim == 3, bhw_tensor.shape
    assert num_classes > bhw_tensor.max(), torch.unique(bhw_tensor)
    # bhw,c
    one_hot = torch.eye(num_classes).index_select(dim=0, index=bhw_tensor.reshape(-1))
    one_hot = one_hot.reshape(*bhw_tensor.shape, num_classes)
    return one_hot


if __name__ == "__main__":
    a = torch.load("/home/lart/Coding/SODBetterProj/tools/a.pt")

    start = time.time()
    b = bhw_to_onehot_by_for(a, num_classes=88).float()
    print("bhw_to_onehot_by_for", time.time() - start)
    start = time.time()
    b1 = bhw_to_onehot_by_for_V1(a, num_classes=88).float()
    print("bhw_to_onehot_by_for_V1", time.time() - start)
    start = time.time()
    c = bhw_to_onehot_by_scatter(a, num_classes=88).float()
    print("bhw_to_onehot_by_scatter", time.time() - start)
    start = time.time()
    c1 = bhw_to_onehot_by_scatter_V1(a, num_classes=88).float()
    print("bhw_to_onehot_by_scatter_V1", time.time() - start)
    start = time.time()
    d = bhw_to_onehot_by_index_select(a, num_classes=88).float()
    print("bhw_to_onehot_by_index_select", time.time() - start)
    start = time.time()
    e = one_hot(a, num_classes=88).float()
    print("one_hot", time.time() - start)
    print(torch.all(torch.isclose(b, b1)))
    print(torch.all(torch.isclose(b, c)))
    print(torch.all(torch.isclose(b, c1)))
    print(torch.all(torch.isclose(b, d)))
    print(torch.all(torch.isclose(b, e)))
    # 0.31998324394226074
    # 0.03971743583679199
    # 0.028532743453979492
    # tensor(0.) tensor(0.)
