import xtrack as xt
import numpy as np
import pybobyqa

line = xt.Line.from_json('xtrack/test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.build_tracker()

# Match tunes and chromaticities to assigned values
opt = line.match(
    solve=False,
    method='4d', # <- passed to twiss
    vary=[
        xt.VaryList(['kqtf.b1', 'kqtd.b1'], step=1e-8, tag='quad'),
        xt.VaryList(['ksf.b1', 'ksd.b1'], step=1e-4, limits=[-0.1, 0.1], tag='sext'),
    ],
    targets = [
        xt.TargetSet(qx=62.315, qy=60.325, tol=1e-6, tag='tune'),
        xt.TargetSet(dqx=10.0, dqy=12.0, tol=0.01, tag='chrom'),
    ])

merit_function = opt.get_merit_function(return_scalar=False, check_limits=False)

bounds = merit_function.get_x_limits()
x0 = merit_function.get_x()

opt.target_status()

#import pybobyqa
#soln = pybobyqa.solve(merit_function, x0=x0,
#            bounds=bounds.T, # wants them transposed...
#            rhobeg=5e-4, rhoend=1e-9, # 1e-4 creates the bug (err_values), 5e-4 converges
#            objfun_has_noise=True, # <-- helps in this case
#            seek_global_minimum=True)
#soln.x

import scipy
soln = scipy.optimize.root(merit_function, x0)

print(soln.success, soln.message)

print(soln.x)

merit_function.set_x(soln.x)

opt.target_status()