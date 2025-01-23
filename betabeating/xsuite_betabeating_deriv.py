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
    env.new('dquad', 'Multipole', knl=[0., 'dk1l'], at=0.),
])

opt = line.match(
    method='4d',
    solve=False,
    vary=xt.Vary('kq', step=1e-4),
    targets=xt.Target('qx', 0.25, tol=1e-6),
)
opt.solve()
# opt.target_status()
# opt.vary_status()

def cot(x):
    return 1/np.tan(x)

def arccot(x):
    if x < 0:
        return np.arctan(1/x) + np.pi
    return np.arctan(1/x)

def evaluate_phi_bar_B(beta_A, phi_A, phi_B, k1):
    # Define the formula for phi_B_bar
    #phi_bar_B = np.arctan2(1, np.cos(phi_B - phi_A)/np.sin(phi_B - phi_A) - k1 * beta_A) + phi_A
    phi_bar_B = arccot(np.cos(phi_B - phi_A)/np.sin(phi_B - phi_A) - k1 * beta_A) + phi_A
    return phi_bar_B

def evaluate_alpha_bar_B(alpha_B, beta_A, phi_A, phi_B, k1):
    # Define the formula for alpha_B_bar
    phi_bar_B = evaluate_phi_bar_B(beta_A, phi_A, phi_B, k1)
    alpha_bar_B = cot(phi_bar_B - phi_A) - (np.cos(phi_B - phi_A)*np.sin(phi_B - phi_A) - alpha_B * np.sin(phi_B - phi_A)**2)\
                                            /np.sin(phi_bar_B - phi_A)**2
    return alpha_bar_B

def evaluate_beta_bar_B(beta_A, beta_B, phi_A, phi_B, k1):
    # Define the formula for beta_B_bar
    phi_bar_B = evaluate_phi_bar_B(beta_A, phi_A, phi_B, k1)
    beta_bar_B_root = np.sqrt(beta_B) * np.sin(phi_B - phi_A) / np.sin(phi_bar_B - phi_A)
    return beta_bar_B_root**2

def deriv_phi_b_bar(beta_A, phi_A, phi_B, k1l):
    # Define the derivative of phi_B_bar with respect to k1l
    dphi = beta_A / (1 + (cot(phi_B - phi_A) - k1l * beta_A)**2)
    return dphi

def deriv_beta_b_bar(beta_A, beta_B, phi_A, phi_B, k1l):
    # Define the derivative of beta_B_bar with respect to k1l
    phi_B_bar = evaluate_phi_bar_B(beta_A, phi_A, phi_B, k1l)
    dbeta = beta_B * np.sin(phi_B - phi_A)**2 * (-2 * np.cos(phi_B_bar - phi_A)) / np.sin(phi_B_bar - phi_A)**3 * deriv_phi_b_bar(beta_A, phi_A, phi_B, k1l)
    #dbeta = -1 * beta_A * beta_B * np.sin(phi_B - phi_A)**2 * (cot(phi_B - phi_A) - k1l * beta_A)
    return dbeta

def deriv_alpha_b_bar(alpha_B, beta_A, phi_A, phi_B, k1l):
    # Define the derivative of alpha_B_bar with respect to k1l
    dalpha = 2 * beta_A * (cot(phi_B - phi_A) - k1l * beta_A) * \
                (np.cos(phi_B - phi_A) * np.sin(phi_B - phi_A) - alpha_B * np.sin(phi_B - phi_A)**2) - beta_A
    return dalpha

tw0 = line.twiss4d()

alpha_a = tw0.alfx[0]
alpha_b = tw0.alfx[-1]
beta_a = tw0.betx[0]
beta_b = tw0.betx[-1]
phi_a = 2*np.pi*tw0.mux[0]
phi_b = 2*np.pi*tw0.mux[-1]

dk1l = env['dk1l']
step = 1e-9
env['dk1l'] += step
dk1l = env['dk1l']

tw = line.twiss4d(init=tw0)
#alpha_a_new = tw.alfx[0]
alpha_b_new = tw.alfx[-1]
#beta_a_new = tw.betx[0]
beta_b_new = tw.betx[-1]
#phi_a_new = 2*np.pi*tw.mux[0]
phi_b_new = 2*np.pi*tw.mux[-1]

# tw0.plot()
# tw.plot()

dalphab = (alpha_b_new - alpha_b) / step
dbetab = (beta_b_new - beta_b) / step
dphib = (phi_b_new - phi_b) / step

# Print all parameters in a readable way
print(f"alpha_a = {alpha_a}\t\t\t beta_a = {beta_a}\t\t\t phi_a = {phi_a}")
print(f"alpha_b = {alpha_b}\t\t\t beta_b = {beta_b}\t\t\t phi_b = {phi_b}")
print(f"alpha_b_new = {alpha_b_new}\t\t beta_b_new = {beta_b_new}\t\t\t phi_b_new = {phi_b_new}")
print(f"alpha_b_form = {evaluate_alpha_bar_B(alpha_b, beta_a, phi_a, phi_b, dk1l)}\t \
        beta_b_form = {evaluate_beta_bar_B(beta_a, beta_b, phi_a, phi_b, dk1l)}\t \
        phi_b_form = {evaluate_phi_bar_B(beta_a, phi_a, phi_b, dk1l)}")

dbetabform = deriv_beta_b_bar(beta_a, beta_b, phi_a, phi_b, dk1l)
dphibform = deriv_phi_b_bar(beta_a, phi_a, phi_b, dk1l)
dalphabform = deriv_alpha_b_bar(alpha_b, beta_a, phi_a, phi_b, dk1l)

print(dphib, dphibform)
print(dbetab, dbetabform)
print(dalphab, dalphabform)

x = np.linspace(-0.1, 0.1, 10000)
#phi_val = np.array([evaluate_phi_bar_B(beta_a, phi_a, phi_b, xi) for xi in x])
#phi_deriv = np.array([deriv_phi_b_bar(beta_a, phi_a, phi_b, xi) for xi in x])

beta_val = np.array([evaluate_beta_bar_B(beta_a, beta_b, phi_a, phi_b, xi) for xi in x])
beta_deriv_ana = np.array([deriv_beta_b_bar(beta_a, beta_b, phi_a, phi_b, xi) for xi in x])

plt.plot(x, beta_deriv_ana)
plt.show()