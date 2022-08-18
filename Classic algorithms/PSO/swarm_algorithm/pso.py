from particle import Particle
import numpy as np
import matplotlib.pyplot as plt


class PSO:
    def __init__(self, function, dimensions, bounds, population, iterations, mode):
        self.function = function
        self.bounds = bounds
        self.log = []
        self.solution = []
        self.value = 0
        self.mode = mode

        initial_fitness = float("inf") if mode == 'MIN' else -float("inf")
        best_fitness = initial_fitness
        best_fitness_position = []

        swarm = []

        for i in range(population):
            swarm.append(Particle(dimensions, bounds, initial_fitness, mode))

        for i in range(iterations):
            for j in range(population):
                swarm[j].eval(function)

                if mode == 'MIN':
                    if swarm[j].position_fitness < best_fitness:
                        best_fitness_position = list(swarm[j].position)
                        best_fitness = float(swarm[j].position_fitness)
                if mode == 'MAX':
                    if swarm[j].position_fitness > best_fitness:
                        best_fitness_position = list(swarm[j].position)
                        best_fitness = float(swarm[j].position_fitness)
            for j in range(population):
                swarm[j].update_velocity(best_fitness_position)
                swarm[j].update_position(bounds)

            self.log.append(best_fitness)

        self.solution = best_fitness_position
        self.value = best_fitness

    def plot_evolution(self):
        print('Solution:', self.solution)
        print('Function value:', self.value)
        plt.figure(1)
        plt.plot(self.log)
        plt.draw()
        plt.show()

    def plot_evaluated_function(self, res):
        _x = _y = np.linspace(start=self.bounds[0][0], stop=self.bounds[0][1], num=res)
        _X, _Y = np.meshgrid(_x, _y)
        _Z = self.function([_X, _Y])

        plt.figure(2)
        ax = plt.axes(projection='3d')
        ax.plot_surface(_X, _Y, _Z)
        ax.scatter(self.solution[0], self.solution[1], self.value, c="#ff0000", marker='X')
        if self.mode == 'MIN':
            ax.set_title('Function minimum')
        else:
            ax.set_title('Function maximum')
        plt.draw()
