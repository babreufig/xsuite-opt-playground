import xtrack as xt
import numpy as np
import matplotlib.pyplot as plt

env = xt.Environment()
env.particle_ref = xt.Particles(p0c=7e12)

env['kq'] = 0.1
env['dk1l'] = 0.

line = env.new_line(components=[
    env.new('qf', 'Quadrupole', k1='kq', length=1.0, anchor='start', at=0.),
    env.new('qd', 'Quadrupole', k1='-kq', length=1.0, anchor='start', at=10.,
            from_='qf@end'),
    env.new('end', 'Marker', at=10., from_='qd@end'),
    #env.new('dquad', 'Multipole', knl=[0., 'dk1l'], at=0),
])

opt = line.match(
    method='4d',
    solve=False,
    vary=xt.Vary('kq', step=1e-4),
    targets=xt.Target('qx', 0.25, tol=1e-6),
)
opt.solve()
opt.target_status()
opt.vary_status()

tw0 = line.twiss4d()

alpha_a = tw0.alfx[0]
alpha_b = tw0.alfx[-1]
beta_a = tw0.betx[0]
beta_b = tw0.betx[-1]
phi_a = 2*np.pi*tw0.mux[0]
phi_b = 2*np.pi*tw0.mux[-1]
kq = env['kq']

opt = line.match(
    method='4d',
    solve=False,
    vary=xt.Vary('kq', step=1e-4),
    targets=xt.Target('mux', 0.2465, at='end', tol=1e-6)#xt.Target('betx', 35.8, at='end', tol=1e-6),
)

merit_function = opt.get_merit_function(check_limits=False)

def cot(x):
    return 1/np.tan(x)

def deriv_phi_b_bar(beta_A, phi_A, phi_B, k1l):
    # Define the derivative of phi_B_bar with respect to k1l
    dphi = beta_A / (1 + (cot(phi_B - phi_A) - k1l * beta_A)**2)
    return dphi

def deriv_beta_b_bar(beta_A, phi_A, phi_B, k1l):
    # Define the derivative of beta_B_bar with respect to k1l
    dbeta = -2 * (cot(phi_B - phi_A) - k1l * beta_A) * deriv_phi_b_bar(beta_A, phi_A, phi_B, k1l)
    return dbeta

print(tw0.alfx[0], tw0.alfx[-1], tw0.betx[0], tw0.betx[-1], 2*np.pi*tw0.mux[0], 2*np.pi*tw0.mux[-1])

print(merit_function.get_jacobian([kq]))

print(merit_function.merit_function._extract_knob_values())
print(kq)

print(deriv_phi_b_bar(beta_a, phi_a, phi_b, kq))

x = np.linspace(-0.1, 0.1, 10000)
y = np.array([deriv_phi_b_bar(beta_a, phi_a, phi_b, xi) for xi in x])

plt.plot(x, y)
plt.show()

opt.solve()
opt.vary_status()
tw0 = line.twiss4d()
print(tw0.mux[-1]*2*np.pi)