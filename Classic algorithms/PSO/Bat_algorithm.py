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


import numpy as np


class BatAlgorithm(BaseSearchAlgorithm):

    def __init__(self, **kwargs):
        super().__init__("bat", **kwargs)

        self.a_init = self.params['a']  # początkowa głośność nietoperzy
        self.r_max = self.params['r_max']  # maksymalny rytm serca nietoperzy
        self.alpha = self.params['alpha']  # współczynnik spadku głośności
        self.gamma = self.params['gamma']  # współczynnik wzrostu rytmu serca
        self.f_min = self.params['f_min']  # minimalna częstotliwość
        self.f_max = self.params['f_max']  # maksymalna częstotliwość

    def initialize(self):
        self.solutions = np.zeros(shape=(self.n, self.d))
        for i in range(self.n):
            self.solutions[i] = self.random_uniform_in_ranges()

        self.q = np.zeros(self.n)
        self.v = np.zeros((self.n, self.d))
        self.b = np.zeros((self.n, self.d))

        self.a = np.repeat(self.a_init, self.n)
        self.r = np.zeros(self.n)

    def constraints_satisfied(self):
        return self.f_min < self.f_max

    def execute_search_step(self, t):
        self.f = np.random.uniform(self.f_min, self.f_max, self.n)

        for i in range(self.n):

            if np.random.uniform(0, 1) < self.r[i]:
                self.b[i] = self.best_solution + np.mean(self.a) * np.random.uniform(-1, 1, self.d)
            else:
                self.v[i] = self.v[i] + (self.solutions[i] - self.best_solution) * self.f[i]
                self.b[i] = self.solutions[i] + self.v[i]

            self.b[i] = self.clip_to_ranges(self.b[i])

            if self.compare_objective_value(self.b[i], self.solutions[i]) < 0:
                if np.random.uniform(0, 1) < self.a[i]:
                    self.solutions[i] = self.b[i]
                    self.a[i] *= self.alpha
                    self.r[i] = self.r_max * (1 - np.exp(-self.gamma * t))

sphere_fct = lambda x: sum([x[i] ** 2 for i in range(len(x))])
rastrigin_fct = lambda x: 10 * len(x) + sum([x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i]) for i in range(len(x))])

d = 2  # wymiar funkcji
n = 100  # rozmiar populacji nietoperzy
range_min, range_max = -5.0, 5.0  # zakres, w którym poszukiwane jest rozwiązanie
T = 200  # liczba powtórzeń

bat = BatAlgorithm(d=d, n=n, range_min=range_min, range_max=range_max,
                   a=0.5, r_min=0.7, r_max=1.0, alpha=0.9, gamma=0.9, f_min=0.0, f_max=5.0)

objective = 'min' # określenie czy poszukiwane jest minimum lub maximum ('min' , 'max')
objective_fct = sphere_fct # testowana funkcja

solution, latency = bat.search(objective, objective_fct, T)
print(solution)

print("czas poszukiwań: ", latency)

objective = 'min'# określenie czy poszukiwane jest minimum lub maximum ('min' , 'max')
objective_fct = rastrigin_fct # testowana funkcja

solution, latency = bat.search(objective, objective_fct, T)
print(solution)
print("czas poszukiwań: ", latency)
