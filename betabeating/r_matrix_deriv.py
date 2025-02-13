import xtrack as xt
import numpy as np
import sympy as sp

sp.init_printing()

env = xt.Environment()
env.particle_ref = xt.Particles(p0c=7e12)

env['kq'] = 0.1
env['dk1l'] = 0.

line = env.new_line(components=[
    env.new('qf', 'Quadrupole', k1='kq', length=1.0, anchor='start', at=0.),
    env.new('qd', 'Quadrupole', k1='-kq', length=1.0, anchor='start', at=10.,
            from_='qf@end'),
    env.new('end', 'Marker', at=10., from_='qd@end'),
    env.new('dquad', 'Multipole', knl=[0., 'dk1l'], at=0),
])

opt = line.match(
    method='4d',
    solve=False,
    vary=xt.Vary('kq', step=1e-4),
    targets=xt.Target('qx', 0.166666, tol=1e-6),
)

opt.solve()
opt.target_status()
opt.vary_status()

tw0 = line.twiss4d()

beta_A = tw0.betx[0]
beta_B = tw0.betx[-1]
alpha_A = tw0.alfx[0]
alpha_B = tw0.alfx[-1]
mu_A = tw0.mux[0]
mu_B = tw0.mux[-1]
dx_A = tw0.dx[0]
dpx_A = tw0.dpx[0]

bety_B = tw0.bety[-1]
alfy_B = tw0.alfy[-1]
muy_B = tw0.muy[-1]

R_AB = tw0.get_R_matrix(tw0.name[0], tw0.name[-1])


# Define Variables R11, R12, R21 and R22 with Sympy
R11, R12, R21, R22, x, b1, a1, mu1, R11t, R12t, R21t, R22t, dx1, dpx1 = sp.symbols('R11 R12 R21 R22 x b1 a1 mu1 R11t R12t R21t R22t dx1 dpx1')
phi1 = 2 * sp.pi * mu1

R_AB_sp = sp.Matrix([[R11, R12], [R21, R22]])
R_Q_sp = sp.Matrix([[1, 0], [-x, 1]])
R_AB_tld = R_AB_sp @ R_Q_sp

R_AB_2 = R_AB_tld @ sp.Matrix([dx1, dpx1])
print(R_AB_2)

R11t, R12t, R21t, R22t = R_AB_tld[0, 0], R_AB_tld[0, 1], R_AB_tld[1, 0], R_AB_tld[1, 1]

# Define formula for propagation
b2 = 1/b1 * ((R11t * b1 - R12t * a1)**2 + R12t**2)
a2 = -1/b1 * ((R11t * b1 - R12t * a1) * (R21t * b1 - R22t * a1) + R12t * R22t)
mu2 = mu1 + sp.atan(R12t / (R11t * b1 - R12t * a1))
phi2 = 2 * sp.pi * mu2

db2 = b2.diff(x).simplify()
da2 = a2.diff(x).simplify()
dmu2 = mu2.diff(x).simplify()
dphi2 = phi2.diff(x).simplify()
dx2 = R_AB_2[0]
dpx2 = R_AB_2[1]

ddx2 = dx2.diff(x).simplify()
ddpx2 = dpx2.diff(x).simplify()

dk1l = 1e-6
env['dk1l'] = dk1l
tw_dev = line.twiss4d(init=tw0)
R_ABt = tw_dev.get_R_matrix(tw0.name[0], tw0.name[-1])

dbeta_num = (tw_dev.betx[-1] - beta_B) / dk1l
dalpha_num = (tw_dev.alfx[-1] - alpha_B) / dk1l
dphi_num = (2*np.pi*(tw_dev.mux[-1] - mu_B)) / dk1l

print(dbeta_num)

# Calculate value for db2 with k = 1e-6
db2_val = db2.subs({R11: R_AB[0, 0], R12: R_AB[0, 1], R21: R_AB[1, 0], R22: R_AB[1, 1], x: dk1l, b1: beta_A, a1: alpha_A, mu1: mu_A})
print(db2_val)


da2_val = da2.subs({R11: R_AB[0, 0], R12: R_AB[0, 1], R21: R_AB[1, 0], R22: R_AB[1, 1], x: dk1l, b1: beta_A, a1: alpha_A, mu1: mu_A})
print(da2_val)
print(dalpha_num)

dphi2_val = dphi2.subs({R11: R_AB[0, 0], R12: R_AB[0, 1], R21: R_AB[1, 0], R22: R_AB[1, 1], x: dk1l, b1: beta_A, a1: alpha_A, mu1: mu_A})
print(dphi2_val)
print(dphi_num)

dmu2_val = dmu2.subs({R11: R_AB[0, 0], R12: R_AB[0, 1], R21: R_AB[1, 0], R22: R_AB[1, 1], x: dk1l, b1: beta_A, a1: alpha_A, mu1: mu_A})
print(dmu2_val)
print(dphi_num)

ddx2_val = ddx2.subs({R11: R_AB[0, 0], R12: R_AB[0, 1], R21: R_AB[1, 0], R22: R_AB[1, 1], x: dk1l, b1: beta_A, a1: alpha_A, mu1: mu_A, dx1: dx_A, dpx1: dpx_A})
print(ddx2_val)
print(tw_dev.dx[-1])

ddpx2_val = ddpx2.subs({R11: R_AB[0, 0], R12: R_AB[0, 1], R21: R_AB[1, 0], R22: R_AB[1, 1], x: dk1l, b1: beta_A, a1: alpha_A, mu1: mu_A, dx1: dx_A, dpx1: dpx_A})
print(ddpx2_val)
print(tw_dev.dpx[-1])

dbety_num = (tw_dev.bety[-1] - bety_B) / dk1l
dalfy_num = (tw_dev.alfy[-1] - alfy_B) / dk1l
dmuy_num = 2 * np.pi * (tw_dev.muy[-1] - muy_B) / dk1l

print(dbety_num)

print(dalfy_num)

print(dmuy_num)