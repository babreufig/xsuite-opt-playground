# Optics calculation and matching for a large ring (LHC) - part 1

import xtrack as xt
import lattice_data.lhc_match as lm

import numpy as np
import pandas as pd
import json
from util.constants import HLLHC15_THICK_PATH, OPT_150_1500_PATH

# Load LHC model
collider = xt.Multiline.from_json(HLLHC15_THICK_PATH)
collider.vars.load_madx_optics_file(OPT_150_1500_PATH)

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

# Match for target bety: 0.15 --> [0.1, 0.14, 0.149, 0.1499, 0.15]

opt.target_status()

knob_values_old = np.array(list(opt.get_knob_values().values())) # Get old knob values

rows = []

import pybobyqa
from scipy.optimize import least_squares, Bounds

#for bety in [0.1, 0.14, 0.149, 0.1499, 0.15]:
for bety in [0.15]:
    print("---------------------------------------------------------------------")
    print(f"Solving for bety: 0.15 --> {bety}")

    opt.targets[7].value = bety # Set bety target

    # print("PyBOBYQA")
    # merit_function = opt.get_merit_function(return_scalar=True, check_limits=False)
    # bounds = merit_function.get_x_limits()
    # x0 = merit_function.get_x()
    # merit_function.merit_function.call_counter = 0

    # soln = pybobyqa.solve(merit_function, x0=x0,
    #             bounds=bounds.T, # wants them transposed...
    #             rhobeg=1e-6, rhoend=1e-16, maxfun=4000, # set maximum to 4000 function evaluations
    #             objfun_has_noise=True, # <-- helps in this case
    #             seek_global_minimum=True)
    # merit_function.set_x(soln.x)
    # opt.target_status()

    # knob_values_new = opt._err._x_to_knobs(merit_function.get_x())
    # knob_values_diff = knob_values_new - knob_values_old

    # row = {'Algorithm': 'PyBOBYQA', 'Target bety': bety, 'Successful': opt._err.last_point_within_tol, '# Calls': merit_function.merit_function.call_counter,
    #        'Penalty': merit_function(soln.x), 'Avg Change knob': np.sqrt(np.mean(knob_values_diff**2)),'Max Change knob': np.max(knob_values_diff)}
    # rows.append(row)
    # opt.tag(f"PyBOBYQA-{bety}")

    # opt.reload(0)
    # print("Least Squares: Trust Region Reflective Algorithm")
    # merit_function = opt.get_merit_function(return_scalar=False, check_limits=False)
    # bounds = merit_function.get_x_limits()
    # x0 = merit_function.get_x()
    # merit_function.merit_function.call_counter = 0
    # soln = least_squares(merit_function, x0, bounds=Bounds(bounds.T[0], bounds.T[1]),
    #                      xtol=1e-12, jac=merit_function.merit_function.get_jacobian, max_nfev=500, verbose=2)
    # merit_function.set_x(soln.x)
    # opt.target_status()

    # knob_values_new = opt._err._x_to_knobs(merit_function.get_x())
    # knob_values_diff = knob_values_new - knob_values_old

    # row = {'Algorithm': 'LSQ-TRF', 'Target bety': bety, 'Successful': opt._err.last_point_within_tol, '# Calls': merit_function.merit_function.call_counter,
    #        'Penalty': np.dot(merit_function(soln.x), merit_function(soln.x)), 'Avg Change knob': np.sqrt(np.mean(knob_values_diff**2)),'Max Change knob': np.max(knob_values_diff)}
    # rows.append(row)
    # opt.tag(f"LS-TRF-{bety}")

    # opt.reload(0)
    # print("Least Squares: Dogbox Algorithm")
    # merit_function = opt.get_merit_function(return_scalar=False, check_limits=False)
    # bounds = merit_function.get_x_limits()
    # x0 = merit_function.get_x()
    # merit_function.merit_function.call_counter = 0
    # soln = least_squares(merit_function, x0, bounds=Bounds(bounds.T[0], bounds.T[1]),
    #                      xtol=1e-12, method='dogbox', jac=merit_function.merit_function.get_jacobian, max_nfev=500)
    # merit_function.set_x(soln.x)
    # opt.target_status()

    # knob_values_new = opt._err._x_to_knobs(merit_function.get_x())
    # knob_values_diff = knob_values_new - knob_values_old

    # row = {'Algorithm': 'LS-Dogbox', 'Target bety': bety, 'Successful': opt._err.last_point_within_tol, '# Calls': merit_function.merit_function.call_counter,
    #        'Penalty': np.dot(merit_function(soln.x), merit_function(soln.x)), 'Avg Change knob': np.sqrt(np.mean(knob_values_diff**2)),'Max Change knob': np.max(knob_values_diff)}
    # rows.append(row)
    # opt.tag(f"LS-Dogbox-{bety}")

    # opt.reload(0)
    # print("Jacobian Method (Xsuite)")
    # merit_function = opt.get_merit_function(return_scalar=False, check_limits=False)
    # bounds = merit_function.get_x_limits()
    # x0 = merit_function.get_x()
    # merit_function.merit_function.call_counter = 0
    # opt.solve()
    # opt.target_status()

    # knob_values_new = opt._err._x_to_knobs(merit_function.get_x())
    # knob_values_diff = knob_values_new - knob_values_old

    # row = {'Algorithm': 'Jacobian', 'Target bety': bety, 'Successful': opt._err.last_point_within_tol, '# Calls': merit_function.merit_function.call_counter,
    #        'Penalty': opt.solver.penalty_after_last_step**2, 'Avg Change knob': np.sqrt(np.mean(knob_values_diff**2)),'Max Change knob': np.max(knob_values_diff)}
    # rows.append(row)
    # opt.tag(f"Xsuite-Jacobian-{bety}")

    opt.reload(0)
    print("L-BFGS-B")
    opt._err.call_counter = 0
    try:
        #opt.run_l_bfgs_b(1)
        opt.run_l_bfgs_b(1200)
    except ValueError:
        print("L-BFGS-B failed")
        n_calls = opt._err.call_counter
        row = {'Algorithm': 'L-BFGS-B', 'Target bety': bety, 'Successful': False, '# Calls': n_calls,
            'Penalty': 1e200, 'Avg Change knob': 0.0, 'Max Change knob': 0.0}
    else:
        n_calls = opt._err.call_counter
        opt.target_status()

        x_sol = opt._err._get_x()

        opt.target_status()
        knob_values_new = opt._err._x_to_knobs(x_sol)
        knob_values_diff = knob_values_new - knob_values_old

        row = {'Algorithm': 'L-BFGS-B', 'Target bety': bety, 'Successful': opt._err.last_point_within_tol, '# Calls': n_calls,
            'Penalty': np.dot(opt._err(x_sol), opt._err(x_sol)), 'Avg Change knob': np.sqrt(np.mean(knob_values_diff**2)),
            'Max Change knob': np.max(knob_values_diff), 'X': opt.get_knob_values()}
        opt.tag(f"Xsuite-lbfgs-b-{bety}")

    rows.append(row)

    opt.reload(0)
    print("Simplex")
    opt._err.call_counter = 0
    #opt.run_simplex(2)
    opt.run_simplex(10000)
    n_calls = opt._err.call_counter
    opt.target_status()

    x_sol = opt._err._get_x()

    opt.target_status()
    knob_values_new = opt._err._x_to_knobs(x_sol)
    knob_values_diff = knob_values_new - knob_values_old

    row = {'Algorithm': 'Nelder-Mead', 'Target bety': bety, 'Successful': opt._err.last_point_within_tol, '# Calls': n_calls,
           'Penalty': np.dot(opt._err(x_sol), opt._err(x_sol)), 'Avg Change knob': np.sqrt(np.mean(knob_values_diff**2)),
           'Max Change knob': np.max(knob_values_diff), 'X': opt.get_knob_values()}
    rows.append(row)
    opt.tag(f"Xsuite-simplex-b-{bety}")

    opt.reload(0)
    # print("BFGS")
    # opt._err.call_counter = 0
    # try:
    #     opt.run_bfgs(600, xrtol=1e-24, gtol=1e-24)
    # except ValueError:
    #     print("BFGS failed")
    #     n_calls = opt._err.call_counter
    #     row = {'Algorithm': 'BFGS', 'Target bety': bety, 'Successful': False, '# Calls': n_calls,
    #         'Penalty': 1e200, 'Avg Change knob': 0.0, 'Max Change knob': 0.0}
    # else:
    #     n_calls = opt._err.call_counter
    #     opt.target_status()

    #     x_sol = opt._err._get_x()

    #     opt.target_status()
    #     knob_values_new = opt._err._x_to_knobs(x_sol)
    #     knob_values_diff = knob_values_new - knob_values_old

    #     row = {'Algorithm': 'BFGS', 'Target bety': bety, 'Successful': opt._err.last_point_within_tol, '# Calls': n_calls,
    #         'Penalty': np.dot(opt._err(x_sol), opt._err(x_sol)), 'Avg Change knob': np.sqrt(np.mean(knob_values_diff**2)),
    #         'Max Change knob': np.max(knob_values_diff), 'X': opt.get_knob_values()}
    #     opt.tag(f"Xsuite-bfgs-{bety}")

    rows.append(row)

opt.log()

# df = pd.DataFrame(rows)
# print(df)
# df.to_csv("results_scalar.csv", index=False)
# with open("results_scalar.json", "w") as f:
#     json.dump(rows, f)