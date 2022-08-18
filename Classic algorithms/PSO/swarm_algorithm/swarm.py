from pso import PSO
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D


def banana(x):
    a = 1. - x[0]
    b = x[1] - x[0] * x[0]
    return a * a + b * b * 100.


def eggholder(x):
    X = x[0]
    Y = x[1]

    return (-(Y + 47.0) * np.sin(np.sqrt(abs(X / 2.0 + (Y + 47.0)))) - X * np.sin(np.sqrt(abs(X - (Y + 47.0)))))


def objective_function(x):
    y = 3 * (1 - x[0]) ** 2 * math.exp(-x[0] ** 2 - (x[1] + 1) ** 2) - 10 * (
            x[0] / 5 - x[0] ** 3 - x[1] ** 5) * math.exp(-x[0] ** 2 - x[1] ** 2) - 1 / 3 * math.exp(
        -(x[0] + 1) ** 2 - x[1] ** 2);
    return y


particle_swarm = PSO(eggholder, 2, [(-512, 512), (-512, 512)], 200, 400, 'MAX')
particle_swarm.plot_evaluated_function(200)
particle_swarm.plot_evolution()
