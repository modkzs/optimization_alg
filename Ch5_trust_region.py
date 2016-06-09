# -*- coding: utf-8 -*-
from common.Function import Rosenbrock
import numpy as np
import math
import matplotlib.pyplot as pl

__author__ = 'yixuanhe'


def trust_region(delta_max, eta, solve, target, init=None, iteration=10000):
    delta = delta_max
    error = []
    if init is None:
        x = np.random.uniform(0, 1, 2)
    else:
        x = init

    for i in range(iteration):
        g = target.derivative(x)
        B = target.second_derivative(x)
        p = solve(g, B, delta)
        if np.dot(p, p) == 0:
            error.append(0)
            continue
        rho = (target.func(x + p) - target.func(x)) / (np.dot(g, p) + np.dot(np.dot(p, B), p)/2)
        if rho < 0.25:
            delta *= 0.25
        elif rho > 0.75 and np.dot(p, p) == eta:
            delta = min(2*delta, delta_max)

        if rho > eta:
            x = x + p
        else:
            continue
        error.append(np.dot(np.array(x)-np.array([1, 1]), np.array(x)-np.array([1, 1])))
        print(x)
        print(delta)
    return error


def cauchy_point(g, B, delta):
    l2_g = np.dot(g, g)

    p = - delta/math.sqrt(l2_g) * g
    l = np.dot(np.dot(g, B), g)
    if l <= 0:
        return p
    elif l2_g**(3/2) > l*delta:
        return p
    else:
        return (l2_g**(3/2)/(l*delta))*p


def dogleg(g, B, delta):
    p_B = -np.dot(np.array(np.mat(B).I), g)

    if np.dot(p_B, p_B) < delta*delta:
        return p_B

    p_U = -(np.dot(g, g)/np.dot(np.dot(g, B), g))*g
    if np.dot(p_U, p_U) > delta*delta:
        return delta/math.sqrt(np.dot(p_U, p_U)) * p_U
    else:
        p = p_B - p_U
        a = np.dot(p, p)
        b = 2*np.dot(p, p_U)
        c = np.dot(p_U, p_U) - delta*delta

        s = math.sqrt(-4*a*c + b*b)
        x = (s - b)/(2*a)
        if x < 1:
            return p_U + x*p
        else:
            x = (s - b)/(2*a)
            return p_U + x*p


r1 = trust_region(1, 0.2, dogleg, Rosenbrock, [1.2, 1.2], 1000)
r2 = trust_region(1, 0.2, cauchy_point, Rosenbrock, [1.2, 1.2], 1000)
length = min(len(r1), len(r2))
x = [i for i in range(length)]

pl.plot(x, r1[:length])
pl.plot(x, r2[:length])
pl.show()
