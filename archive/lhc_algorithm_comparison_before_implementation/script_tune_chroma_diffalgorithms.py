import xtrack as xt
import pybobyqa
from scipy.optimize import least_squares, Bounds
import pandas as pd
import numpy as np
from util.constants import LHC_THICK_KNOBS_PATH

# Load a line and build a tracker
line = xt.Line.from_json(LHC_THICK_KNOBS_PATH)
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

rows = []

print("PyBOBYQA")
merit_function = opt.get_merit_function(return_scalar=True, check_limits=True)
bounds = merit_function.get_x_limits()
x0 = merit_function.get_x()
merit_function.merit_function.call_counter = 0

soln = pybobyqa.solve(merit_function, x0=x0,
            bounds=bounds.T, # wants them transposed...
            maxfun=4000, # set maximum to 4000 function evaluations
            rhobeg=5e-4, rhoend=1e-9, objfun_has_noise=True, # <-- helps in this case
            seek_global_minimum=True)
merit_function.set_x(soln.x)
opt.target_status()

row = {'Algorithm': 'PyBOBYQA', 'Successful': opt._err.last_point_within_tol, '# Calls': merit_function.merit_function.call_counter,
        'Penalty': merit_function(soln.x)}
rows.append(row)
opt.tag(f"PyBOBYQA")
print(soln)

opt.reload(0)
print("Least Squares: Trust Region Reflective Algorithm")
merit_function = opt.get_merit_function(return_scalar=False, check_limits=True)
bounds = merit_function.get_x_limits()
x0 = merit_function.get_x()
merit_function.merit_function.call_counter = 0
soln = least_squares(merit_function, x0, bounds=Bounds(bounds.T[0], bounds.T[1]),
                        xtol=1e-12, jac=merit_function.merit_function.get_jacobian, max_nfev=500)
merit_function.set_x(soln.x)
opt.target_status()

row = {'Algorithm': 'LSQ-TRF', 'Successful': opt._err.last_point_within_tol, '# Calls': merit_function.merit_function.call_counter,
        'Penalty': np.dot(merit_function(soln.x), merit_function(soln.x))}
rows.append(row)
opt.tag(f"LS-TRF")
print(soln)

opt.reload(0)
print("Least Squares: Dogbox Algorithm")
merit_function = opt.get_merit_function(return_scalar=False, check_limits=False)
bounds = merit_function.get_x_limits()
x0 = merit_function.get_x()
merit_function.merit_function.call_counter = 0
soln = least_squares(merit_function, x0, bounds=Bounds(bounds.T[0], bounds.T[1]),
                        xtol=1e-12, method='dogbox', jac=merit_function.merit_function.get_jacobian, max_nfev=500)
merit_function.set_x(soln.x)
opt.target_status()

row = {'Algorithm': 'LS-Dogbox', 'Successful': opt._err.last_point_within_tol, '# Calls': merit_function.merit_function.call_counter,
        'Penalty': np.dot(merit_function(soln.x), merit_function(soln.x))}
rows.append(row)
opt.tag(f"LS-Dogbox")

opt.reload(0)
print("Xsuite method")
merit_function = opt.get_merit_function(return_scalar=False, check_limits=False)
bounds = merit_function.get_x_limits()
x0 = merit_function.get_x()
merit_function.merit_function.call_counter = 0
opt.solve()
opt.target_status()

row = {'Algorithm': 'Jacobian', 'Successful': opt._err.last_point_within_tol, '# Calls': merit_function.merit_function.call_counter,
        'Penalty': np.dot(merit_function(soln.x), merit_function(soln.x))}
rows.append(row)
opt.tag(f"Xsuite method")

df = pd.DataFrame(rows)
print(df)