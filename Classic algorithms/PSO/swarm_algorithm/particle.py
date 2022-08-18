import random


class Particle:
    def __init__(self, dimensions, bounds, initial_fitness, mode='MAX', inertia=0.85, cognitive=1, social=2):
        self.dimensions = dimensions
        self.mode = mode
        self.position = []
        self.velocity = []
        self.best_position = []
        self.best_position_fitness = initial_fitness
        self.position_fitness = initial_fitness

        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

        for i in range(self.dimensions):
            self.position.append(random.uniform(bounds[i][0], bounds[i][1]))
            self.velocity.append(random.uniform(-1, 1))

    def eval(self, function):
        self.position_fitness = function(self.position)

        if self.mode == 'MIN':
            if self.position_fitness < self.best_position_fitness:
                self.best_position = self.position
                self.best_position_fitness = self.position_fitness
        if self.mode == 'MAX':
            if self.position_fitness > self.best_position_fitness:
                self.best_position = self.position
                self.best_position_fitness = self.position_fitness

    def update_velocity(self, best_particle_position):
        for i in range(self.dimensions):
            r1 = random.random()
            r2 = random.random()

            cognitive_velocity = self.cognitive * r1*(self.best_position[i] - self.position[i])
            social_velocity = self.social * r2*(best_particle_position[i] - self.position[i])
            self.velocity[i] = self.inertia * self.velocity[i] + cognitive_velocity + social_velocity

    def update_position(self, bounds):
        for i in range(self.dimensions):
            self.position[i] = self.position[i] + self.velocity[i]

            if self.position[i] > bounds[i][1]:
                self.position[i] = bounds[i][1]

            if self.position[i] < bounds[i][0]:
                self.position[i] = bounds[i][0]
