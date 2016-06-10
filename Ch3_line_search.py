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


def line_search_with_wolfe(target, init=None, alpha0=1, phi=0.5, c_1=0.001, c_2=0.9):
    if init is None:
        x = np.random.uniform(0, 1, 2)
    else:
        x = init

    i = 0
    error = []
    last_alpha = None
    while i < 10000:
        p = -target.derivative(x)
        alpha = 0
        while True:
            cur = x + alpha*p
            if i > 1:
                if target.func(cur) > target.func(x) + c_1*alpha*np.dot(target.derivative(x), p) \
                        or target.func(cur) > target.func(x + last_alpha*p):
                    alpha = zoom([last_alpha, alpha], x, p, target, c_1, c_2)
                    break
            elif target.func(cur) > target.func(x) + c_1*alpha*np.dot(target.derivative(x), p):
                alpha = zoom([last_alpha, alpha], x, p, target, c_1, c_2)
                break

            d_a = np.dot(target.derivative(cur), p)
            d_0 = np.dot(target.derivative(x), p)

            if abs(d_a) <= c_2*d_0:
                break
            if d_a >= 0:
                alpha = zoom([alpha, last_alpha], x, p, target, c_1, c_2)
                break
            last_alpha = alpha
            alpha = (alpha+alpha0)/2

        print(alpha)
        print(x)
        x += alpha*p
        error.append(np.dot(x-np.array([1, 1]), x-np.array([1, 1])))
        i += 1
    return error


def zoom(alpha, x, p, target, c_1, c_2):
    while True:
        begin = x + alpha[0]*p
        end = x + alpha[1]*p
        cur_alpha = (alpha[0] + alpha[1])/2

        cur = x + cur_alpha*p

        if target.func(cur) > target.func(x) + c_1*cur_alpha*np.dot(target.derivative(x), p) \
                or target.func(cur) > target.func(begin):
            alpha[1] = cur_alpha
        else:
            d_a = np.dot(target.derivative(cur), p)
            d_0 = np.dot(target.derivative(x), p)
            if abs(d_a) <= -c_2*d_0:
                return cur_alpha
            elif d_a*(alpha[1]-alpha[0]) >= 0:
                alpha[1] = alpha[0]

            alpha[0] = cur_alpha

if __name__ == "__main__":
    r = line_search_with_wolfe(Rosenbrock, init=[1.2, 1.2])
    x = [i for i in range(len(r))]

    pl.plot(x, r)
    pl.show()

