import xtrack as xt
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import approx_fprime
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

RESCALE = True
if RESCALE:
    merit_function = opt.get_merit_function(return_scalar=True, check_limits=False, rescale_x=(0,1))
    bounds = merit_function.get_x_limits()
    x0 = merit_function.get_x()

else:
    merit_function = opt.get_merit_function(return_scalar=True, check_limits=False)
    bounds = [(-1e-4, 1e-4), (-1e-4, 1e-4), (-0.1, 0.1), (-0.1, 0.1)]
    print(merit_function.get_x_limits())
    x0 = merit_function.get_x()

opt.check_limits = False

#bounds = merit_function.get_x_limits()

def status_query(intermediate_result):
    if isinstance(intermediate_result, np.ndarray):
        x = intermediate_result
    else:
        x = intermediate_result.x
    merit_function.set_x(x)
    opt.target_status()

#x0 = merit_function.get_x()
#x0 = np.array([0.5,0.5,0.5,0.5])

#x0 = [ 4.2633e-05, -4.2895e-05] = solution
# Use L-BFGS-B with user-provided gradient

if RESCALE:
    result = minimize(merit_function, x0=x0, jac=merit_function.get_jacobian, method='BFGS', callback=status_query)
else:
    result = minimize(merit_function, x0=x0, bounds=bounds, method='L-BFGS-B', callback=status_query)
print("Optimal solution:", result.x)
print("Objective value:", result.fun)

merit_function.set_x(result.x)
opt.target_status()