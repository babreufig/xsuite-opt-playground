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



start = "ip2"
end = "ip7"

#collider.lhcb1.cycle(start)

# Twiss on a part of the machine (bidirectional)
tw_81_12 = collider.lhcb1.twiss(start=start, end=end, init_at=start,
                                betx=0.15, bety=0.15)

# add multipole at IP8 for calculating derivatives
dk1l = 0.01

collider.lhcb1.insert(collider.new('dquad', 'Multipole', knl=[0., dk1l]), from_=start, at=start)
print(collider.lhcb1['dquad'].knl)
tw_81_12_new = collider.lhcb1.twiss(start=start, end=end, init_at=start,
                                betx=0.15, bety=0.15)

# %%
# opt = collider.lhcb1.match(
#     solve=False,
#     default_tol={None: 1e-8, 'betx': 1e-6, 'bety': 1e-6, 'alfx': 1e-6, 'alfy': 1e-6},
#     start='s.ds.l8.b1', end='ip1',
#     init=tw0, init_at=xt.START,
#     vary=[
#         # Only IR8 quadrupoles including DS
#         xt.VaryList(['kq6.l8b1', 'kq7.l8b1', 'kq8.l8b1', 'kq9.l8b1', 'kq10.l8b1',
#             'kqtl11.l8b1', 'kqt12.l8b1', 'kqt13.l8b1',
#             'kq4.l8b1', 'kq5.l8b1', 'kq4.r8b1', 'kq5.r8b1',
#             'kq6.r8b1', 'kq7.r8b1', 'kq8.r8b1', 'kq9.r8b1',
#             'kq10.r8b1', 'kqtl11.r8b1', 'kqt12.r8b1', 'kqt13.r8b1'])],
#     targets=[
#         xt.TargetSet(at='ip8', tars=('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx'), value=tw0),
#         xt.TargetSet(at='ip1', betx=0.15, bety=0.10, alfx=0, alfy=0, dx=0, dpx=0),
#         xt.TargetRelPhaseAdvance('mux', value = tw0['mux', 'ip1.l1'] - tw0['mux', 's.ds.l8.b1']),
#         xt.TargetRelPhaseAdvance('muy', value = tw0['muy', 'ip1.l1'] - tw0['muy', 's.ds.l8.b1']),
#     ])

def finite_diff(line, k, start, end, eps=1e-4):
    line['dquad'].knl = [0, k]
    tw0 = line.twiss(init_at=start, start=start, end=end, betx=0.15, bety=0.15)
    line['dquad'].knl = [0, k + eps]
    tw1 = line.twiss(init=tw0, start=start, end=end)
    return (tw1.bety[-1] - tw0.bety[-1]) / eps, tw1

derivs = tw_81_12_new.get_twiss_param_derivative(start, end)
diffs, tw_table = finite_diff(collider.lhcb1, dk1l, start, end)
print(derivs)
print(diffs)