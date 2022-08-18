import numpy as np

from time import process_time


class BaseSearchAlgorithm():

    def __init__(self, name, **kwargs):
        print(name, kwargs)
        self.name = name
        self.objective = None
        self.objective_fct = None

        self.solutions = []
        self.history = []
        self.best_solution = None
        self.params = kwargs

        self.n = self.params['n']  # rozmiar populacji
        self.d = self.params['d']  # wymiar, w którym operuje funkcja

        self.range_min = self.params['range_min']  # dolna granica poszukiwań
        self.range_max = self.params['range_max']  # górna granica poszukiwań

        if np.isscalar(self.range_min):
            self.range_min = np.repeat(self.range_min, self.d)

        if np.isscalar(self.range_max):
            self.range_max = np.repeat(self.range_max, self.d)

        self.evaluation_count = 0

    def constraints_satisfied(self):
        return True

    def get_best_solution(self, key=None):
        if not key:
            key = self.objective_fct

        if self.objective == 'min':
            candidate = min(self.solutions, key=key)
        elif self.objective == 'max':
            candidate = max(self.solutions, key=key)

        if self.best_solution is None or self.compare_objective_value(candidate, self.best_solution) < 0:
            self.best_solution = np.copy(candidate)

        return self.best_solution

    def compare_objective_value(self, s0, s1):
        v0 = self.objective_fct(s0)
        v1 = self.objective_fct(s1)

        if self.objective == 'min':
            return v0 - v1
        elif self.objective == 'max':
            return v1 - v0

    def argsort_objective(self):
        if self.objective == 'min':
            return np.argsort([self.objective_fct(s) for s in self.solutions]).ravel()
        elif self.objective == 'max':
            return np.argsort([self.objective_fct(s) for s in self.solutions])[::-1].ravel()

    def evaluation_count_decorator(self, f, x):
        self.evaluation_count += 1
        return f(x)

    def random_uniform_in_ranges(self):
        rnd = np.zeros(self.d)
        for i in range(self.d):
            rnd[i] = np.random.uniform(self.range_min[i], self.range_max[i])
        return rnd

    def clip_to_ranges(self, x):
        for i in range(self.d):
            x[i] = np.clip(x[i], self.range_min[i], self.range_max[i])
        return x

    def search(self, objective, objective_fct, T):

        if not self.constraints_satisfied():
            return (np.nan, np.nan), np.nan

        self.objective = objective
        self.evaluation_count = 0
        self.objective_fct = lambda x: self.evaluation_count_decorator(objective_fct, x)
        self.history = np.zeros((T, self.d))
        self.best_solution = self.random_uniform_in_ranges()
        self.initialize()

        t_start = process_time()

        for t in range(T):
            self.execute_search_step(t)
            self.history[t] = self.get_best_solution()

        t_end = process_time()

        return (self.best_solution, self.objective_fct(self.best_solution)), t_end - t_start

    def initialize(self):
        raise NotImplementedError

    def execute_search_step(self, t):
        raise NotImplementedError


class BeesAlgorithm(BaseSearchAlgorithm):

    def __init__(self, **kwargs):
        super().__init__("bees", **kwargs)

        self.nb = self.params['nb']  # liczba pszczół zwiadowców
        self.ne = self.params['ne']  # liczba elitarnych pszczół zwiadowców
        self.nrb = self.params['nrb']  # liczba rekrutowanych pszczół przez zwiadowców
        self.nre = self.params['nre']  # liczba rekrutowancyh pszczół przez elitanrych zwiadowców

        self.initial_size = 1.0

    def constraints_satisfied(self):
        return self.ne <= self.nb

    def initialize(self):
        self.solutions = np.zeros((self.n, self.d))
        self.flower_patch = [None] * self.n
        self.size = [self.initial_size] * self.n

        for i in range(self.n):
            self.initialize_flower_patch(i)

    def execute_search_step(self, t):

        self.waggle_dance()

        for i in range(self.nb):
            self.local_search(i)

        for i in range(self.nb, self.n):
            self.global_search(i)

    def initialize_flower_patch(self, i):
        self.solutions[i] = self.create_random_scout()
        self.flower_patch[i] = {'size': self.initial_size}

    def create_random_scout(self):
        return self.random_uniform_in_ranges()

    def create_random_forager(self, i):
        nght = self.flower_patch[i]['size']
        forager = np.random.uniform(-1, 1) * nght + self.solutions[i]
        for j in range(self.d):
            forager[j] = np.clip(forager[j], self.range_min[j], self.range_max[j])
        return forager

    def waggle_dance(self):
        idxs = self.argsort_objective()
        self.solutions = self.solutions[idxs]
        self.flower_patch = np.array(self.flower_patch)[idxs].ravel()


    def local_search(self, i):
        for j in range(self.nrb if i < self.nb else self.nre):
            forager = self.create_random_forager(i)
            if self.compare_objective_value(forager, self.solutions[i]) < 0:
                self.solutions[i] = forager
                self.initialize_flower_patch(i)

    def global_search(self, i):
        self.initialize_flower_patch(i)


sphere_fct = lambda x: sum([x[i] ** 2 for i in range(len(x))])
rastrigin_fct = lambda x: 10 * len(x) + sum([x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i]) for i in range(len(x))])

d = 3  # wymiar funkcji
n = 100  # rozmiar populacji pszczół
range_min, range_max = -5.0, 5.0  # zakres, w którym poszukiwane jest rozwiązanie
T = 100  # liczba powtórzeń

bees = BeesAlgorithm(d=d, n=n, range_min=range_min, range_max=range_max,
                     nb=50, ne=20, nrb=5, nre=10, shrink_factor=0.8, stgn_lim=5)

objective = 'min' # określenie czy poszukiwane jest minimum lub maximum ('min' , 'max')
objective_fct = sphere_fct # testowana funkcja

solution, latency = bees.search(objective, objective_fct, T)
print(solution)

print("czas poszukiwań: ", latency)

objective = 'min'# określenie czy poszukiwane jest minimum lub maximum ('min' , 'max')
objective_fct = rastrigin_fct # testowana funkcja

solution, latency = bees.search(objective, objective_fct, T)
print(solution)
print("czas poszukiwań: ", latency)
