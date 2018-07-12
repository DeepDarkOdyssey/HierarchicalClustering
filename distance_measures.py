# -*- coding: utf-8 -*-

import numpy as np
from numba import jit


def find_lcslen(s1, s2, c):
    """查找两个序列s1，s2的最长公共子序列的长度

    :param s1: list or str
    :param s2: list or str
    :param c: list of list, 用来存储动态规划结果的矩阵
    :return: int, 两个序列s1和s2的最长公共子序列的长度
    """
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                c[i+1][j+1] = c[i][j] + 1
            elif c[i+1][j] > c[i][j+1]:
                c[i+1][j+1] = c[i+1][j]
            else:
                c[i+1][j+1] = c[i][j+1]

    return c[len(s1)][len(s2)]

@jit
def find_lcslen_np(s1, s2, c):
    """查找两个序列s1，s2的最长公共子序列的长度，使用了numpy和numba.jit加速

    :param s1: list or str
    :param s2: list or str
    :param c: numpy.ndarray, 用来存储动态规划结果的矩阵
    :return: int, 两个序列s1和s2的最长公共子序列的长度
    """
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                c[i+1, j+1] = c[i, j] + 1
            elif c[i+1, j] > c[i, j+1]:
                c[i+1, j+1] = c[i+1, j]
            else:
                c[i+1, j+1] = c[i, j+1]

    return c[len(s1), len(s2)]


def lcs_distance(s1, s2):
    """计算两个序列s1，s2之间的距离度量

    :param s1: list or str
    :param s2: list or str
    :return: 0~1的float，代表序列s1和s2之间的距离，距离最短为0,最大为1
    """
    # c = [[0 for x in range(len(s2)+1)] for y in range(len(s1)+1)]
    c = np.zeros((len(s1)+1, len(s2)+1), dtype=np.int)
    lcs_len = find_lcslen_np(s1, s2, c)
    return 1 - (lcs_len / min(len(s1), len(s2)))     # 最好为0，最差为1


def jaccard_distance(s1, s2):
    """计算两个序列s1，s2之间的距离度量

    :param s1: list or str
    :param s2: list or str
    :return: 0~1的float，代表序列s1和s2之间的距离，距离最短为0,最大为1
    """
    # token_set1 = set(word_segment_own(s1))
    # token_set2 = set(word_segment_own(s2))
    token_set1 = set(s1)
    token_set2 = set(s2)
    return 1 - len(token_set1.intersection(token_set2)) / len(token_set1.union(token_set2))
