from xdeps.optimize import optimize as opti
import xtrack as xt
import numpy as np

def f(x):
    return [x[0], x[0]**2 + x[1]**2 - 4]

x0 = [-0.5,1.2]
steps = [1e-8, 1e-8]
tar = [-1., 0.]
tols = [1e-6, 1e-6, 1e-6]

vary = [xt.Vary(ii, container=x0, step=steps[ii]) for ii in range(len(x0))]


opt = opti.Optimize(vary=vary, targets=xt.match.ActionCall(f, vary).get_targets(tar))

for ii, tt in enumerate(opt.targets):
        tt.tol = tols[ii]

opt.solve()
opt.target_status()
print(opt.get_knob_values())