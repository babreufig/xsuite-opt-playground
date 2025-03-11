from xdeps.optimize import optimize as opti
import xtrack as xt
import numpy as np

def f(x):
    return [x[0], x[0]**2 + x[1]**2, x[0]**2 + x[1]**2]

x0 = [1.0,0.0]
steps = [1e-8, 1e-8]
tar = [-2., xt.match.LessThan(4.0), xt.match.GreaterThan(3.9)]
tols = [1e-8, 1e-8, 1e-8]

vary = [xt.Vary(ii, container=x0, step=steps[ii]) for ii in range(len(x0))]

# Doesn't work anymore, ActionCall is not there

opt = opti.Optimize(vary=vary, targets=xt.match.ActionCall(f, vary).get_targets(tar))
opt.targets[0].weight = 2

for ii, tt in enumerate(opt.targets):
        tt.tol = tols[ii]

it = 0
for i in range(opt.n_steps_max):
    old_x = opt.solver.x if opt.solver.x is not None else x0.copy()
    if opt.solver.stopped is not None:
        break
    it += 1
    opt.solver.x = opt._err._knobs_to_x(opt._extract_knob_values())
    opt.step(1)
    opt.set_knobs_from_x(opt.solver.x)
    knobs = opt._extract_knob_values()

    print(f"Iteration {it}: ({np.round(old_x[0], 3)}, {np.round(old_x[1], 3)}) " + \
          f"---> ({np.round(knobs[0], 3)}, {np.round(knobs[1], 3)})")

opt.target_status()