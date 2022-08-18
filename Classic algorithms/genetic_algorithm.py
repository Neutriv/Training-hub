import numpy

class Genetic:
    def optimization(self, generations, equation_inputs, num_weights, sol_per_pop, num_parents_mating):
        pop_size = (sol_per_pop,num_weights)
        new_population = numpy.random.uniform(low=-4.0, high=4.0, size=pop_size)
        print(new_population)

        for generation in range(generations):
            print("Generation : ", generation)
            # Ocena jakości
            fitness = self.__cal_pop_fitness(equation_inputs, new_population)

            # Wybór najlepszych rodziców
            parents = self.__select_mating_pool(new_population, fitness, num_parents_mating)

            # Krzyzowanie
            offspring_crossover = self.__crossover(parents, offspring_size=(pop_size[0]-parents.shape[0], num_weights))

            # Mutacja
            offspring_mutation = self.__mutation(offspring_crossover)

            # Selekcja
            new_population[0:parents.shape[0], :] = parents
            new_population[parents.shape[0]:, :] = offspring_mutation

            print("Best result : ", numpy.max(numpy.sum(new_population*equation_inputs, axis=1)))

        fitness = self.__cal_pop_fitness(equation_inputs, new_population)
        best_match_idx = numpy.where(fitness == numpy.max(fitness))

        print("Best solution : ", new_population[best_match_idx, :])
        print("Best solution fitness : ", fitness[best_match_idx])

    def __cal_pop_fitness(self, equation_inputs, pop):
        fitness = numpy.sum(pop*equation_inputs, axis=1)
        return fitness

    def __select_mating_pool(self, pop, fitness, num_parents):
        parents = numpy.empty((num_parents, pop.shape[1]))
        for parent_num in range(num_parents):
            max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
            max_fitness_idx = max_fitness_idx[0][0]
            parents[parent_num, :] = pop[max_fitness_idx, :]
            fitness[max_fitness_idx] = -99999999999
        return parents

    def __crossover(self, parents, offspring_size):
        offspring = numpy.empty(offspring_size)
        crossover_point = numpy.uint8(offspring_size[1]/2)

        for k in range(offspring_size[0]):
            parent1_idx = k%parents.shape[0]
            parent2_idx = (k+1)%parents.shape[0]
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        return offspring

    def __mutation(self, offspring_crossover, num_mutations=1):
        mutations_counter = numpy.uint8(offspring_crossover.shape[1] / num_mutations)
        for idx in range(offspring_crossover.shape[0]):
            gene_idx = mutations_counter - 1
            for mutation_num in range(num_mutations):
                random_value = numpy.random.uniform(-1.0, 1.0, 1)
                offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
                gene_idx = gene_idx + mutations_counter
        return offspring_crossover

"""
y = w1x1+w2x2+w3x3+w4x4+w5x5+6wx6
gdzie (x1,x2,x3,x4,x5,x6)=(2,-1,3,5,-2.5,2.4)
"""
inputs = [2,-1,3,5,-2.5,2.4]
weights = 6
sol_per_pop = 8
parents_mating = 4

genetic = Genetic()
genetic.optimization(5, inputs, weights, sol_per_pop, parents_mating)
