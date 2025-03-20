import xtrack as xt
import lattice_data.lhc_match as lm
import numpy as np
from util.constants import HLLHC15_THICK_PATH, OPT_150_1500_PATH
from tabulate import tabulate

env = xt.Environment()
env.particle_ref = xt.Particles(p0c=7e12)

# Load LHC model

collider = env.from_json(HLLHC15_THICK_PATH)
collider.vars.load_madx(OPT_150_1500_PATH)

collider.build_trackers()

line = collider.lhcb1

start = "ip2"
end = "ip7"

dk1l = 0.1
line.insert(collider.new('dquad', 'Multipole', knl=[0., dk1l]), from_=start, at=start)

tw0 = line.twiss()
tw_lim = line.twiss(init=tw0, start=start, end=end)

derivs = tw0.get_twiss_param_derivative(src=start, observation=end)

eps = 1e-2

line['dquad'].knl[1] += eps
tw_plus = line.twiss(init=tw_lim, start=start, end=end)
line['dquad'].knl[1] -= 2 * eps
tw_minus = line.twiss(init=tw_lim, start=start, end=end)
line['dquad'].knl[1] += eps

eps_small = 1e-4

line['dquad'].knl[1] += eps_small
tw_plus_smalleps = line.twiss(init=tw_lim, start=start, end=end)
line['dquad'].knl[1] -= 2 * eps_small
tw_minus_smalleps = line.twiss(init=tw_lim, start=start, end=end)
line['dquad'].knl[1] += eps_small

fd_dict = {
    'dbetx': (tw_plus.betx[-1] - tw_minus.betx[-1]) / (2 * eps),
    'dbety': (tw_plus.bety[-1] - tw_minus.bety[-1]) / (2 * eps),
    'dalfx': (tw_plus.alfx[-1] - tw_minus.alfx[-1]) / (2 * eps),
    'dalfy': (tw_plus.alfy[-1] - tw_minus.alfy[-1]) / (2 * eps),
    'dmux': (tw_plus_smalleps.mux[-1] - tw_minus_smalleps.mux[-1]) / (2 * eps_small),
    'dmuy': (tw_plus_smalleps.muy[-1] - tw_minus_smalleps.muy[-1]) / (2 * eps_small),
    'ddx': (tw_plus.dx[-1] - tw_minus.dx[-1]) / (2 * eps),
    'ddpx': (tw_plus.dpx[-1] - tw_minus.dpx[-1]) / (2 * eps),
    'ddy': (tw_plus.dy[-1] - tw_minus.dy[-1]) / (2 * eps),
    'ddpy': (tw_plus.dpy[-1] - tw_minus.dpy[-1]) / (2 * eps),
}

def compare_dicts(d1, d2):
    headers = ["Parameter", "Twiss Derivative", "FD Derivative", "Absolute Difference", "Relative Difference"]
    table = []

    for key in d1.keys():
        val1 = d1[key]
        val2 = d2[key]
        abs_diff = abs(val1 - val2)
        rel_diff = (abs_diff / abs(val1)) if val1 != 0 else float('inf')

        table.append([key, f"{val1:.6g}", f"{val2:.6g}", f"{abs_diff:.6g}", f"{rel_diff:.6g}"])

    print(tabulate(table, headers, tablefmt="fancy_grid"))

compare_dicts(derivs, fd_dict)

# Assert that finite differences and twiss derivatives are the same
for key in derivs.keys():
    atol = 1e-8
    if key == 'dbetx' or key == 'dbety':
        rtol = 1e-2
    elif key in ['ddx', 'ddpx', 'ddy', 'ddpy']:
        rtol = 0
        atol = 1e-7
    else:
        rtol = 1e-6
    assert np.isclose(derivs[key], fd_dict[key], atol=atol, rtol=rtol),\
        f"Error in {key}. Absolute difference is {abs(derivs[key] - fd_dict[key]):.6g},\
        relative difference is {abs((derivs[key] - fd_dict[key]) / derivs[key]):.6g}"