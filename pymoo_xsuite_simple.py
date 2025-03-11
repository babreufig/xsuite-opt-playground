import xtrack as xt
import numpy as np

from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.sampling.lhs import LHS

from util.constants import LHC_THICK_KNOBS_PATH

# Load a line and build a tracker
line = xt.Line.from_json(LHC_THICK_KNOBS_PATH)
line.build_tracker()

# Match tunes and chromaticities to assigned values
opt = line.match(
    solve=False,
    method='4d',
    vary=[
        xt.VaryList(['kqtf.b1', 'kqtd.b1'], step=1e-8, limits=[-1e-4, 1e-4], tag='quad'),
        xt.VaryList(['ksf.b1', 'ksd.b1'], step=1e-4, limits=[-0.1, 0.1], tag='sext'),
    ],
    targets = [
        xt.TargetSet(qx=62.315, qy=60.325, tol=1e-6, tag='tune'),
        xt.TargetSet(dqx=10.0, dqy=12.0, tol=0.01, tag='chrom'),
    ])


merit_function = opt.get_merit_function(return_scalar=False, check_limits=False)

bounds = merit_function.get_x_limits()
print(bounds)
x0 = merit_function.get_x()

# Need to force certain boundaries if there aren't
# otherwise the space is too large for sampling
#bounds[0] = [-1e-4, 1e-4]
#bounds[1] = [-1e-4, 1e-4]

n_var = 4
n_obj = 4

class MyProblem(Problem):

    def __init__(self, surrogate=None):
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=0,
                         xl=bounds[:,0],  # Lower bounds
                         xu=bounds[:,1])   # Upper bounds
        self.surrogate = surrogate

    def _evaluate(self, x, out, *args, **kwargs):
        if self.surrogate:
            # Use surrogate to predict the objective function values
            out["F"] = self.surrogate.predict(x)
        else:
            out["F"] = np.array([merit_function(x_i)**2 for x_i in x])

problem = MyProblem()

# Population size
pop_size = 50

sample = LHS()
sample_x = sample(problem, pop_size - 1).get("X")

# Generate initial population including the starting point x0
initial_population = np.vstack([x0, sample_x])

# Set up reference directions for 4 objectives
ref_dirs = get_reference_directions("das-dennis", n_var, n_partitions=2)

# Configure NSGA-III algorithm
algorithm = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
#algorithm = KGB()


# Perform the optimization
res = minimize(problem,
               algorithm,
               termination=('n_gen', 40),
               seed=1,
               save_history=True,
               verbose=True)

# Output results
print("Optimal Solutions (Variables):")
print(res.X)  # Decision variables (kqtf.b1, kqtd.b1, ksf.b1, ksd.b1)
print("Objective Values (Errors):")
print(res.F)  # Objective values (errors in qx, qy, dqx, dqy)


#Scatter().add(res.F).show()

merit_function.set_x(res.X[np.argmin(np.sum(res.F**2, axis=1))])
opt.target_status()