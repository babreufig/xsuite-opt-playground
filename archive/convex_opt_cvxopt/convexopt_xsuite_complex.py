from cvxopt import matrix, solvers

import xtrack as xt
import lattice_data.lhc_match as lm

import numpy as np
import pandas as pd
from util.constants import HLLHC15_THICK_PATH, OPT_150_1500_PATH

# Load LHC model

collider = xt.Multiline.from_json(HLLHC15_THICK_PATH)
collider.vars.load_madx_optics_file(OPT_150_1500_PATH)

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
        xt.TargetSet(at='ip1', betx=0.15, bety=0.149, alfx=0, alfy=0, dx=0, dpx=0),
        xt.TargetRelPhaseAdvance('mux', value = tw0['mux', 'ip1.l1'] - tw0['mux', 's.ds.l8.b1']),
        xt.TargetRelPhaseAdvance('muy', value = tw0['muy', 'ip1.l1'] - tw0['muy', 's.ds.l8.b1']),
    ])

merit_function = opt.get_merit_function(return_scalar=False, check_limits=False)

bounds = merit_function.get_x_limits()
x0 = merit_function.get_x()

def F(x=None, z=None):
    if x is None:
        return 14, matrix(merit_function.get_x())

    f = merit_function(x) # Shape (14, 1)
    fwsum = np.sum(f**2).reshape(1)

    f = np.concat((fwsum, f))

    jacobian = merit_function.merit_function.get_jacobian(x)
    jacobian_sum = np.sum(jacobian, axis=0).reshape(1, len(jacobian[0]))
    f_jacobian = np.concat((jacobian_sum, jacobian))

    if z is None:
        return matrix(f), matrix(f_jacobian)
    else:
        # Compute the Lagrangian Hessian, ensuring it's an n x n matrix
        H = z[0] * (f_jacobian.T @ f_jacobian)
        return matrix(f), matrix(f_jacobian), matrix(H)

def F_old(x=None, z=None):
    if x is None:
        return 0, matrix(merit_function.get_x())

    f = merit_function(x) # Shape (20, 1)
    jacobian = merit_function.merit_function.get_jacobian(x)

    f = np.sum(f)
    f_jacobian = np.sum(jacobian, axis=0).reshape(1, len(jacobian[0]))

    if z is None:
        return matrix(f), matrix(f_jacobian)
    else:
        # Compute the Lagrangian Hessian, ensuring it's an n x n matrix
        H = z[0] * (f_jacobian.T @ f_jacobian)
        return matrix(f), matrix(f_jacobian), matrix(H)

# Inequality Constraints:
G = np.zeros((40,20), dtype=float)
for i in range(0, len(G), 2):
    G[i][i//2] = -1.0
    G[i+1][i//2] = 1.0
G = matrix(G)
h = matrix(bounds.flatten())


# Solve
solution = solvers.cp(F, G=G, h=h)

# Optimal solution
x_opt = solution['x']
optimal_value = solution['primal objective']

print("Optimal solution (x):", x_opt[0], x_opt[1])
print("Minimal value of the function:", optimal_value)