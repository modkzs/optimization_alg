# -*- coding: utf-8 -*-
import numpy as np
__author__ = 'yixuanhe'


class Function:
    @staticmethod
    def func(x):
        pass

    @staticmethod
    def derivative(x):
        pass

    @staticmethod
    def second_derivative(x):
        pass


class Rosenbrock(Function):
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
