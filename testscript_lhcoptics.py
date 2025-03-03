# Optics calculation and matching for a large ring (LHC) - part 1

import xtrack as xt
import lhc_match as lm

import numpy as np
import os
import pandas as pd
import json

# Load LHC model

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
collider = xt.Multiline.from_json(dir_path + "/hllhc15_collider_thick.json")
collider.vars.load_madx_optics_file(dir_path + "/opt_round_150_1500.madx")

collider.build_trackers()

# Initial twiss
tw0 = collider.lhcb1.twiss()

# Inspect IPS
tw0.rows['ip.*'].cols['betx bety mux muy x y']


# Prepare for optics matching: set limits and steps for all circuits
lm.set_var_limits_and_steps(collider)

# Inspect for one circuit
collider.vars.vary_default['kq4.l2b2']

# Twiss on a part of the machine (bidirectional)
tw_81_12 = collider.lhcb1.twiss(start='ip8', end='ip2', init_at='ip1',
                                betx=0.15, bety=0.15)


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

opt.check_limits = False

opt.target_status()

knob_values_old = np.array(list(opt.get_knob_values().values())) # Get old knob values

print("---------------------------------------------------------------------")
print(f"Solving for bety: 0.15 --> 0.1")

opt._err.call_counter = 0
opt.solve()
# opt.step(1, broyden=True)
# opt.step(30, broyden=True)
# opt.step(10, broyden=True)
# #opt.step(2, broyden=False)
# #opt.step(5, broyden=True)


n_calls = opt._err.call_counter

#x_sol = opt._err._get_x()

opt.target_status()
# knob_values_new = opt._err._x_to_knobs(x_sol)
# knob_values_diff = knob_values_new - knob_values_old

print(opt.log())