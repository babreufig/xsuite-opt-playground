import xtrack as xt
import lattice_data.lhc_match as lm
from util.constants import HLLHC15_THICK_PATH, OPT_150_1500_PATH
import numpy as np

# Load LHC model
collider = xt.Environment.from_json(HLLHC15_THICK_PATH)
collider.vars.load_madx(OPT_150_1500_PATH)

collider.build_trackers()

line = collider.lhcb1

line.cycle('ip7', inplace=True)

# Initial twiss
tw0 = line.twiss()

# Inspect IPS
tw0.rows['ip.*'].cols['betx bety mux muy x y']


# Prepare for optics matching: set limits and steps for all circuits
lm.set_var_limits_and_steps(collider)

# Inspect for one circuit
collider.vars.vary_default['kq4.l2b2']

# Twiss on a part of the machine (bidirectional)
tw_81_12 = line.twiss(start='ip8', end='ip2', init_at='ip1',
                                betx=0.15, bety=0.15)


opt = line.match(
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

class FakeQuad:
    pass

env = collider

vv = "kq7.l8b1"

k1 = []
myelems = {}
for dd in env.ref_manager.find_deps([env.vars[vv]]):
    if dd.__class__.__name__ == "AttrRef" and dd._key == "k1":
        k1.append((dd._owner._key, dd._expr))
        myelems[dd._owner._key] = FakeQuad()

fdef = env.ref_manager.mk_fun("myfun", a=env.vars[vv])
gbl = {
    "vars": env.ref_manager.containers["vars"]._owner.copy(),
    "element_refs": myelems,
}
lcl = {}
exec(fdef, gbl, lcl)
fff = lcl["myfun"]

import sympy

a = sympy.var("a")
fff(a)
dk1_dvv = {}
for kk, expr in k1:
    dd = gbl["element_refs"][kk].k1.diff(a)
    dk1_dvv[kk] = dd
    print(kk, "k1", expr, dd)

quad_names = [kk for kk, _ in k1]

target_place = 'ip1'

twiss_derivs = {}
for qqnn in quad_names:
    twiss_derivs[qqnn] = tw0.get_twiss_param_derivative(src=qqnn, observation=target_place)

    # Refer to k1 instead of k1l
    for nn in twiss_derivs[qqnn].keys():
        twiss_derivs[qqnn][nn] *= env[qqnn].length

target_quantity = 'betx'

target_quantities = ['betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx']
dtar_dvv = np.zeros(len(target_quantities))

for i, target_quantity in enumerate(target_quantities):
    for qqnn in quad_names:
        dtar_dvv[i] += twiss_derivs[qqnn]['d'+target_quantity] * dk1_dvv[qqnn]

err = opt.get_merit_function()
jj = err.get_jacobian(err.get_x())

print("dtar_dvv", dtar_dvv)
print('jac[6,1]', jj[6,1])

for i in range(len(dtar_dvv)):
    print(f'Quantity: {target_quantities[i]},\tDerivative: {dtar_dvv[i]:.5f},\tJacobian: {jj[6+i,1]:.5f}')