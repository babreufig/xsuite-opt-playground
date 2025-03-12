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
tw0 = collider.lhcb1.twiss()

# Inspect IPS
tw0.rows['ip.*'].cols['betx bety mux muy x y']


# Prepare for optics matching: set limits and steps for all circuits
lm.set_var_limits_and_steps(collider)

# Inspect for one circuit
collider.vars.vary_default['kq4.l2b2']

start = "ip5"
end = "ip6"

#collider.lhcb1.cycle(start)

# Twiss on a part of the machine (bidirectional)
# tw_81_12 = collider.lhcb1.twiss(start=start, end=end, init_at=start,
#                                 betx=0.15, bety=0.15)

# add multipole at IP8 for calculating derivatives
dk1l = 0.1

collider.lhcb1.insert(collider.new('dquad', 'Multipole', knl=[0., dk1l]), from_=start, at=start)
tw_81_12_new = collider.lhcb1.twiss(start=start, end=end, init_at=start,
                                betx=0.15, bety=0.15)

derivs = tw_81_12_new.get_twiss_param_derivative(start, end)

def finite_diff(line, k, start, end, eps=1e-6):
    line['dquad'].knl = [0, k]
    tw0 = line.twiss(init=tw_81_12_new, start=start, end=end)
    line['dquad'].knl = [0, k + eps]
    tw1 = line.twiss(init=tw0, start=start, end=end)
    return (tw1.betx[-1] - tw0.betx[-1]) / eps


diffs = finite_diff(collider.lhcb1, dk1l, start, end)
print(derivs)
print(diffs)