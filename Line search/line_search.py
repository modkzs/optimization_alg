# -*- coding: utf-8 -*-
from common.Function import Function
import numpy as np
import matplotlib.pyplot as pl
__author__ = 'yixuanhe'


class Target(Function):
    @staticmethod
    def func(x):
        return 100*(x[0]**2-x[1])**2 + (1-x[0])**2

    @staticmethod
    def derivative(x):
        return np.array([-2*(1-x[0])-400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)])

    @staticmethod
    def second_derivative(x):
        sd = [[-400*(x[1]-3*x[0]**2)+2, -400*x[0]],
              [-400*x[0], 200]]
        return np.array(sd)

    def size(self):
        return 2


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

r = steepest_descent(Target, init=[-1.2, 1])
x = [i for i in range(len(r))]

pl.plot(x, r)
pl.show()

