import pygmo as pg
import numpy as np

class Circle_Problem:

    def fitness(self, x):
        obj = x[0]
        ci1 = -x[0]**2 - x[1]**2 + 3
        ci2 = x[0]**2 + x[1]**2 - 4

        return [x[0], ci1, ci2]

    def get_nic(self):
        return 2 # Number of inequality constraints

    def get_nec(self):
        return 0 # Number of equality constraints

    def get_bounds(self):
        return ([-8]*2, [8]*2)

    def get_nobj(self):
        return 1 # Number of objectives

    def gradient(self, x):
        return ([1, 0, -2*x[0], -2*x[1], 2*x[0], 2*x[1]])

udp = Circle_Problem()
prob = pg.problem(udp)
algo = pg.algorithm(uda = pg.nlopt('auglag'))
algo.extract(pg.nlopt).local_optimizer = pg.nlopt('var2')
algo.set_verbosity(200)
pop = pg.population(prob = udp, size = 1)
pop = algo.evolve(pop)

print(pop.champion_f)
print(pop.champion_x)