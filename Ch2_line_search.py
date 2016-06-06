# -*- coding: utf-8 -*-
from common.Function import Rosenbrock
import numpy as np
import matplotlib.pyplot as pl
__author__ = 'yixuanhe'


def steepest_descent(target, init=None, alpha0=1, phi=0.5, c=0.001):
    if init is None:
        x = np.random.uniform(0, 1, 2)
    else:
        x = init

    i = 0
    alphas = []
    while i < 100:
        alpha = alpha0
        p = -target.derivative(x)
        while target.func(x+alpha*p) > target.func(x) + c*alpha*np.dot(p, p):
            alpha *= phi
        print(alpha)
        print(x)
        x += alpha*p
        d = np.array(x)-np.array([1, 1])
        alphas.append(np.dot(d, d))
        i += 1
    return alphas


def newton_method(target, init=None, alpha0=1, phi=0.5, c=0.001):
    if init is None:
        x = np.random.uniform(0, 1, 2)
    else:
        x = init

    i = 0
    alphas = []
    while i < 100:
        alpha = alpha0
        p = -np.dot(np.array(np.mat(target.second_derivative(x)).I), target.derivative(x))
        while target.func(x+alpha*p) > target.func(x) + c*alpha*np.dot(p, p):
            alpha *= phi
        print(alpha)
        print(x)
        x += alpha*p
        d = np.array(x)-np.array([1, 1])
        alphas.append(np.dot(d, d))
        i += 1
    return alphas


def line_search_with_wolfe(target, init=None, alpha0=1, phi=0.5, c_1=0.001, c_2=0.9, strict = False):
    if init is None:
        x = np.random.uniform(0, 1, 2)
    else:
        x = init

    i = 0
    alphas = []
    while i < 10000:
        alpha = alpha0
        p = -target.derivative(x)
        flag = True
        while flag:
            flag = False
            if target.func(x+alpha*p) > target.func(x) + c_1*alpha*np.dot(p, p):
                alpha *= phi
                flag = True
            if np.dot(target.derivative(x+alpha*p), p) <= c_2*np.dot(target.derivative(x), p):
                alpha *= 2
                flag = True
            if abs(np.dot(target.derivative(x+alpha*p), p)) >= abs(c_2*np.dot(target.derivative(x), p)):
                alpha *= 2
                flag = True

        print(alpha)
        print(x)
        x += alpha*p
        alphas.append(np.dot(x-np.array(1, 1), x-np.array(1, 1)))
        i += 1
    return alphas


def zoom(alpha, x, p, target, c_1):
    while True:
        begin = x + alpha[0]*p
        end = x + alpha[1]*p
        cur_alpha = alpha[0] + alpha[1]/2

        cur = x + cur_alpha*p

        if target.func(cur) > target.func(x) + c_1*cur_alpha*np.dot(target.derivative(begin), p) \
                or target.func(cur) > target.func(begin):
            alpha[1] = cur_alpha
        else:
            d_a = np.dot(target.derivative(cur), p)
            d_0 = np.dot(target.derivative(x), p)
            if abs(d_a) <= c_1*d_0:
                return cur_alpha
            elif d_a*(alpha[1]-alpha[0]) >= 0:
                alpha[1] = alpha[0]

            alpha[0] = cur_alpha

r = steepest_descent(Rosenbrock, init=[-1.2, 1])
x = [i for i in range(len(r))]

pl.plot(x, r)
pl.show()

