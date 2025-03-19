import xtrack as xt
import lattice_data.lhc_match as lm

import numpy as np
from util.constants import HLLHC15_THICK_PATH, OPT_150_1500_PATH

env = xt.Environment()
env.particle_ref = xt.Particles(p0c=7e12)

# Load LHC model

collider = env.from_json(HLLHC15_THICK_PATH)
collider.vars.load_madx(OPT_150_1500_PATH)

collider.build_trackers()

# Initial twiss
line = collider.lhcb1

# Inspect IPS
#tw0.rows['ip.*'].cols['betx bety mux muy x y']


# Prepare for optics matching: set limits and steps for all circuits
# lm.set_var_limits_and_steps(collider)

# Inspect for one circuit
# collider.vars.vary_default['kq4.l2b2']

start = "ip1"
end = "ip3"

#collider.lhcb1.cycle(start)

# Twiss on a part of the machine (bidirectional)
# tw_81_12 = collider.lhcb1.twiss(start=start, end=end, init_at=start,
#                                 betx=0.15, bety=0.15)

# add multipole at IP8 for calculating derivatives
dk1l = 0.1

line.insert(collider.new('dquad', 'Multipole', knl=[0., dk1l]), from_=start, at=start)

tw0 = line.twiss()
tw_lim = line.twiss(init=tw0, start=start, end=end)

derivs = tw0.get_twiss_param_derivative(start, end)

eps = 1e-6

line['dquad'].knl[1] += eps
tw_plus = line.twiss(init=tw_lim, start=start, end=end)
line['dquad'].knl[1] -= 2 * eps
tw_minus = line.twiss(init=tw_lim, start=start, end=end)
line['dquad'].knl[1] += eps

fd_dict = {
    'dbetx': (tw_plus.betx[-1] - tw_minus.betx[-1]) / (2 * eps),
    'dbety': (tw_plus.bety[-1] - tw_minus.bety[-1]) / (2 * eps),
    'dalfx': (tw_plus.alfx[-1] - tw_minus.alfx[-1]) / (2 * eps),
    'dalfy': (tw_plus.alfy[-1] - tw_minus.alfy[-1]) / (2 * eps),
    'dmux': (tw_plus.mux[-1] - tw_minus.mux[-1]) / (2 * eps),
    'dmuy': (tw_plus.muy[-1] - tw_minus.muy[-1]) / (2 * eps),
    'ddx': (tw_plus.dx[-1] - tw_minus.dx[-1]) / (2 * eps),
    'ddpx': (tw_plus.dpx[-1] - tw_minus.dpx[-1]) / (2 * eps),
    'ddy': (tw_plus.dy[-1] - tw_minus.dy[-1]) / (2 * eps),
    'ddpy': (tw_plus.dpy[-1] - tw_minus.dpy[-1]) / (2 * eps),
}
import pprint
pprint.pprint("Twiss Derivatives")
pprint.pprint(derivs)
pprint.pprint("Finite Differences")
pprint.pprint(fd_dict)

# Assert that finite differences and twiss derivatives are the same
for key in derivs.keys():
    assert np.isclose(derivs[key], fd_dict[key], rtol=1e-4, atol=1e-6), f"Error in {key}"