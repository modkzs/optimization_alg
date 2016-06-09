# -*- coding: utf-8 -*-
import numpy as np
import math

__author__ = 'yixuanhe'


def BFGS(s_k, y_k, last=None):
    size = np.array([2, 2]).shape[0]
    i = np.identity(size)
    if last is None:
        return (np.dot(y_k, s_k) / np.dot(y_k, y_k)) * i
    else:
        rhi = 1 / np.dot(y_k, s_k)
        cur = (i - rhi * np.mat(y_k).T * np.mat(s_k)) * np.mat(last) * (i - rhi * np.mat(s_k).T * np.mat(y_k)) \
              + rhi * np.mat(y_k).T * np.mat(y_k)
        return cur


def SR1(s_k, y_k, last=None, r=0.0000001):
    size = np.array([2, 2]).shape[0]
    i = np.identity(size)
    if last is None:
        return (np.dot(y_k, s_k) / np.dot(y_k, y_k)) * i
    elif np.dot(np.mat(y_k) - np.mat(s_k)*np.mat(last).T, np.mat(y_k).T - np.mat(last)*np.mat(s_k).T)[0][0] == 0:
        return last
    elif (np.mat(s_k)*(np.mat(y_k).T-np.mat(last)*np.mat(s_k).T))[0][0] <= r*math.sqrt(np.dot(s_k, s_k)) * \
            math.sqrt(np.array(np.mat(y_k-np.dot(last, s_k))*np.mat(y_k-np.dot(last, s_k)).T)[0][0]):
        return last
    else:
        return last + (np.mat(s_k).T - np.mat(last)*np.mat(y_k).T)*(np.mat(s_k).T - np.mat(last)*np.mat(y_k).T).T /\
                      np.dot(y_k, s_k-np.dot(last, y_k))


last = None
for i in range(1000):
    s_k = np.array([1, 3])
    y_k = np.array([2, 3])
    last = SR1(s_k, y_k, last)
