import numpy as np

from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.visualization.fitness_landscape import FitnessLandscape
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.algorithms.soo.nonconvex.de import DE

rastrigin_problem = get_problem("rastrigin", n_var=14)

#FitnessLandscape(rastrigin_problem, angle=(45, 45), _type="surface").show()

rosenbrock_problem = get_problem("rosenbrock", n_var=20)

#FitnessLandscape(rosenbrock_problem, angle=(45, 45), _type="surface").show()

ref_dirs = get_reference_directions("das-dennis", 1, n_partitions=12)

nsga2 = NSGA2(pop_size=100)
nsga3 = NSGA3(pop_size=100, ref_dirs=np.array([[1.0]]))
cmaes = CMAES(restarts=10, restart_from_best=True, sigma=0.1)
ga = GA(pop_size=100, eliminate_duplicates=True)
de = DE(pop_size=100, jitter=True)
moead = MOEAD(ref_dirs, n_neighbors=15, prob_neighbor_mating=0.7)



res = minimize(rastrigin_problem,
               nsga3,
               ('n_eval', 250000),
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

# plot = Scatter()
# plot.add(rosenbrock_problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
# plot.add(res.F, facecolor="none", edgecolor="red")
# plot.show()
