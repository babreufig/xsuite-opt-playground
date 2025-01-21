from cvxopt import matrix, solvers
import xtrack as xt
import os
import pandas as pd
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

# Load a line and build a tracker
line = xt.Line.from_json(dir_path + '/lhc_thick_with_knobs.json')
line.build_tracker()

# Match tunes and chromaticities to assigned values
opt = line.match(
    solve=False,
    method='4d',
    vary=[
        xt.VaryList(['kqtf.b1', 'kqtd.b1'], step=1e-8, tag='quad'),
        xt.VaryList(['ksf.b1', 'ksd.b1'], step=1e-4, limits=[-0.1, 0.1], tag='sext'),
    ],
    targets = [
        xt.TargetSet(qx=62.315, qy=60.325, tol=1e-6, tag='tune'),
        xt.TargetSet(dqx=10.0, dqy=12.0, tol=0.01, tag='chrom'),
    ])

def twiss_nonlinear_up():
    aa = opt.targets[0].action
    qx = merit_function.merit_function._last_data[aa].qx - opt.targets[0].value - 1e-6
    qy = merit_function.merit_function._last_data[aa].qy - opt.targets[1].value - 1e-6
    dqx = merit_function.merit_function._last_data[aa].dqx - opt.targets[2].value - 0.01
    dqy = merit_function.merit_function._last_data[aa].dqy - opt.targets[3].value - 0.01

    return np.array([qx, qy, dqx, dqy])

merit_function = opt.get_merit_function(return_scalar=False, check_limits=False)

bounds = merit_function.get_x_limits()
x0 = merit_function.get_x()

# To satisfy Gx = h as the constraints
G = np.zeros((len(bounds),len(bounds)), dtype=float)
for i in range(0, len(G), 2):
    G[i][i//2 + 2] = -1.0
    G[i+1][i//2 + 2] = 1.0
G = matrix(G)
h = matrix(bounds.flatten()[4:])

rows = []

def F(x=None, z=None):
    if x is None:
        return 4, matrix(merit_function.get_x())

    f = merit_function(x)
    constr = twiss_nonlinear_up()

    jacobian = merit_function.merit_function.get_jacobian(x)

    f = np.concat(([np.sum(f)], constr))
    f_jacobian = np.sum(jacobian, axis=0).reshape(1, len(jacobian[0]))
    f_jacobian = np.concat((f_jacobian, jacobian))
    

    if z is None:
        return matrix(f), matrix(f_jacobian)
    # Compute the Lagrangian Hessian, ensuring it's an n x n matrix
    H = z[0] * (f_jacobian.T @ f_jacobian) + z[1:5] * (jacobian.T @ jacobian)
    opt.target_status()
    return matrix(f), matrix(f_jacobian), matrix(H)

# Solve
solution = solvers.cp(F, G=G, h=h)

# Optimal solution
x_opt = solution['x']
optimal_value = solution['primal objective']

print("Optimal solution (x):", x_opt[0], x_opt[1])
print("Minimal value of the function:", optimal_value)