import xtrack as xt
import os
import numpy as np
import lhc_match as lm
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.de import DE

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
collider = xt.Multiline.from_json(dir_path + "/hllhc15_collider_thick.json")
collider.vars.load_madx_optics_file(dir_path + "/opt_round_150_1500.madx")

collider.build_trackers()

# Initial twiss
tw0 = collider.lhcb1.twiss()

# Inspect IPS
tw0.rows['ip.*'].cols['betx bety mux muy x y']


# Prepare for optics matching: set limits and steps for all circuits
lm.set_var_limits_and_steps(collider)

# Inspect for one circuit
collider.vars.vary_default['kq4.l2b2']

# Twiss on a part of the machine (bidirectional)
tw_81_12 = collider.lhcb1.twiss(start='ip8', end='ip2', init_at='ip1',
                                betx=0.15, bety=0.15)


# %%
opt = collider.lhcb1.match(
    solve=False,
    default_tol={None: 1e-8, 'betx': 1e-6, 'bety': 1e-6, 'alfx': 1e-6, 'alfy': 1e-6},
    start='s.ds.l8.b1', end='ip1',
    init=tw0, init_at=xt.START,
    vary=[
        # Only IR8 quadrupoles including DS
        xt.VaryList(['kq6.l8b1', 'kq7.l8b1', 'kq8.l8b1', 'kq9.l8b1', 'kq10.l8b1',
            'kqtl11.l8b1', 'kqt12.l8b1', 'kqt13.l8b1',
            'kq4.l8b1', 'kq5.l8b1', 'kq4.r8b1', 'kq5.r8b1',
            'kq6.r8b1', 'kq7.r8b1', 'kq8.r8b1', 'kq9.r8b1',
            'kq10.r8b1', 'kqtl11.r8b1', 'kqt12.r8b1', 'kqt13.r8b1'])],
    targets=[
        xt.TargetSet(at='ip8', tars=('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx'), value=tw0),
        xt.TargetSet(at='ip1', betx=0.15, bety=0.1, alfx=0, alfy=0, dx=0, dpx=0),
        xt.TargetRelPhaseAdvance('mux', value = tw0['mux', 'ip1.l1'] - tw0['mux', 's.ds.l8.b1']),
        xt.TargetRelPhaseAdvance('muy', value = tw0['muy', 'ip1.l1'] - tw0['muy', 's.ds.l8.b1']),
    ])


merit_function = opt.get_merit_function(return_scalar=True, check_limits=False)
bounds = merit_function.get_x_limits()
x0 = merit_function.get_x()

n_var = 20
n_obj = 1

class MyProblem(Problem):

    def __init__(self, surrogate=None):
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=0,
                         xl=bounds[:,0],  # Lower bounds
                         xu=bounds[:,1])   # Upper bounds
        self.surrogate = surrogate

    def _evaluate(self, x, out):
        out["F"] = np.array([merit_function(x_i) for x_i in x])

problem = MyProblem()

# Population size
pop_size = 100
n_eval = 30000

# Use CMAES as an algorithm
algorithm_str = "CMAES"
algorithm = CMAES(restarts=10, restart_from_best=True, sigma=0.1)
#algorithm = DE(pop_size=pop_size, jitter=True)

# Perform the optimization
res = minimize(problem,
               algorithm,
               termination=('n_eval', n_eval),
               seed=1,
               #mutation=PolynomialMutation(prob=1.0 / n_var, eta=20),
               verbose=True)

# Output results
print("Optimal Solutions (Variables):")
print(res.X)
print("Objective Values (Errors):")
print(res.F)

#np.save(f"pymoo_complex_x_{algorithm_str}_neval_{n_eval}_refpart.npy", res.X)
#np.save(f"pymoo_complex_f_{algorithm_str}_neval_{n_eval}_refpart.npy", res.F)

#merit_function.set_x(res.X[np.argmin(np.sum(res.F**2, axis=1))])
merit_function.set_x(res.X)
opt.target_status()