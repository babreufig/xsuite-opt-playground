import xtrack as xt
import varlimits_helper as vlm

import numpy as np
import os

env = xt.Environment()
env.particle_ref = xt.Particles(p0c=7e12)
env['dk1l'] = 0.

# Load LHC model

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

collider = env.from_json(dir_path + "/../hllhc15_collider_thick.json")
collider.vars.load_madx(dir_path + "/../opt_round_150_1500.madx")

collider.build_trackers()

# Initial twiss
tw0 = collider.lhcb1.twiss()

# Inspect IPS
tw0.rows['ip.*'].cols['betx bety mux muy x y']


# Prepare for optics matching: set limits and steps for all circuits
vlm.set_var_limits_and_steps(collider)

# Inspect for one circuit
collider.vars.vary_default['kq4.l2b2']

# Twiss on a part of the machine (bidirectional)
tw_81_12 = collider.lhcb1.twiss(start='ip8', end='ip2', init_at='ip1',
                                betx=0.15, bety=0.15)

# add multipole at IP8 for calculating derivatives

extra_line = env.new_line(components=[env.new('dquad', 'Multipole', knl=[0., 'dk1l'], at='ip8')])

# %%
opt = collider.lhcb1.match(
    solve=False,
    default_tol={None: 1e-8, 'betx': 1e-6, 'bety': 1e-6, 'alfx': 1e-6, 'alfy': 1e-6},
    start='s.ds.l8.b1', end='ip1',
    init=tw0, init_at=xt.START,
    vary=[
        # Only IR8 quadrupoles including DS
        xt.VaryList(['kq6.l8b1', 'kq7.l8b1', 'kq8.l8b1', 'kq9.l8b1', 'kq10.l8b1',
            'kqtl11.l8b1', 'kqt12.l8b1', 'kqt13.l8b1',
            'kq4.l8b1', 'kq5.l8b1', 'kq4.r8b1', 'kq5.r8b1',
            'kq6.r8b1', 'kq7.r8b1', 'kq8.r8b1', 'kq9.r8b1',
            'kq10.r8b1', 'kqtl11.r8b1', 'kqt12.r8b1', 'kqt13.r8b1'])],
    targets=[
        xt.TargetSet(at='ip8', tars=('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx'), value=tw0),
        xt.TargetSet(at='ip1', betx=0.15, bety=0.10, alfx=0, alfy=0, dx=0, dpx=0),
        xt.TargetRelPhaseAdvance('mux', value = tw0['mux', 'ip1.l1'] - tw0['mux', 's.ds.l8.b1']),
        xt.TargetRelPhaseAdvance('muy', value = tw0['muy', 'ip1.l1'] - tw0['muy', 's.ds.l8.b1']),
    ])