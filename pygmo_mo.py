import numpy as np
import pygmo as pg

class Schaffer:
    # Define objectives
    def fitness(self, x):
        f1 = x[0]**2
        f2 = (x[0]-2)**2
        return [f1, f2]

    # Return number of objectives
    def get_nobj(self):
        return 2

    # Return bounds of decision variables
    def get_bounds(self):
        return ([0]*1, [2]*1)

    # Return function name
    def get_name(self):
        return "Schaffer function N.1"

prob = pg.problem(Schaffer())
pop = pg.population(prob, size=200, seed=1)
algo = pg.algorithm(pg.nsga2(gen=40))
pop = algo.evolve(pop)

fits, vectors = pop.get_f(), pop.get_x()

ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(fits)
print(ndf)

print(fits[100], vectors[100])

fit_sums = np.sum(fits, axis=1)
argmin_idx = np.argmin(fit_sums)

print(fits[argmin_idx], vectors[argmin_idx])

import matplotlib.pyplot as plt

# Plotting the Pareto front
plt.scatter(*zip(*fits))
plt.xlabel('Objective 1: f1(x) = x[0]^2')
plt.ylabel('Objective 2: f2(x) = (x[0] - 2)^2')
plt.title('Pareto Front for Schaffer Function N.1')
plt.show()