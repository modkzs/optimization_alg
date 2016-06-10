# -*- coding: utf-8 -*-
import numpy as np
from Ch3_line_search import zoom
import matplotlib.pyplot as pl
from common.Function import Rosenbrock
__author__ = 'yixuanhe'


def conjugate_gradiaent(target, update, init=None, alpha_max=2, c_1=0.001, c_2=0.9):
    if init is None:
        x = np.random.uniform(0, 2, 2)
    else:
        x = init

    d = target.derivative(x)
    p = -d
    errors = []
    while np.dot(d, d) > 0.00000001:
        alpha = 0.5
        i = 1
        last_alpha = 0
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
            alpha = (alpha+alpha_max)/2
            i += 1

        x += alpha*p
        d_c = target.derivative(x)
        beta = update(d_c, d, p)
        p = -d_c + beta*p
        print(x)
        d = d_c
        errors.append(np.dot(x-np.array([1, 1]), x-np.array([1, 1])))
    return errors


def FR(d_c, d, p):
    return np.dot(d_c, d_c)/np.dot(d, d)


def PR(d_c, d, p):
    return np.dot(d_c, d_c-d)/np.dot(d, d)


def PR_plus(d_c, d, p):
    return max(np.dot(d_c, d_c-d)/np.dot(d, d), 0)


def HS(d_c, d, p):
    return np.dot(d_c, d_c-d)/np.dot(d_c-d, p)


def FR_PR(d_c, d, p):
    pr = PR(d_c, d, p)
    fr = FR(d_c, d, p)

    if pr < -fr:
        return -fr
    if abs(pr) <= fr:
        return pr
    if pr > fr:
        return fr

r = conjugate_gradiaent(Rosenbrock, PR, init=[1.2, 1.2])
x = [i for i in range(len(r))]

pl.plot(x, r)
pl.show()
