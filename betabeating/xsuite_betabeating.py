import xtrack as xt
import numpy as np

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
    targets=xt.Target('qx', 0.25, tol=1e-6),
)
opt.solve()
opt.target_status()
opt.vary_status()

tw0 = line.twiss4d()

dk_test = np.linspace(-0.001, 0.001, 100)

mu_test = np.empty_like(dk_test)
beta_test = np.empty_like(dk_test)
alpha_test = np.empty_like(dk_test)

for ii, kk in enumerate(dk_test):
    env['dk1l'] = kk
    mu_test[ii] = line.twiss4d(init=tw0).mux[-1]
    beta_test[ii] = line.twiss4d(init=tw0).betx[-1]
    alpha_test[ii] = line.twiss4d(init=tw0).alfx[-1]

mu0 = tw0.mux[-1]
alf0 = tw0.alfx[-1]
bet0 = tw0.betx[-1]

def evaluate_matrix(alpha_A, alpha_B, beta_A, beta_B, phi_A, phi_B):
    # Define the matrix elements
    element_11 = np.sqrt(beta_B / beta_A) * (np.cos(phi_B - phi_A) + alpha_A * np.sin(phi_B - phi_A))
    element_12 = np.sqrt(beta_B * beta_A) * np.sin(phi_B - phi_A)
    element_21 = ((alpha_A - alpha_B) * np.cos(phi_B - phi_A) - (1 + alpha_A * alpha_B) * np.sin(phi_B - phi_A)) / np.sqrt(beta_B * beta_A)
    element_22 = np.sqrt(beta_A / beta_B) * (np.cos(phi_B - phi_A) - alpha_B * np.sin(phi_B - phi_A))

    # Construct the matrix
    M_AB = np.array([
        [element_11, element_12],
        [element_21, element_22]
    ])

    return M_AB

def evaluate_phi_bar_B(phi_A, phi_B, beta_A, k1):
    # Define the formula for \bar{\phi}_B
    phi_bar_B = np.arctan2(1, np.cos(phi_B - phi_A)/np.sin(phi_B - phi_A) - k1 * beta_A) + phi_A
    return phi_bar_B

def evaluate_beta_bar_B(beta_A, beta_B, phi_A, phi_B, k1):
    # Define the formula for \bar{\beta}_B
    phi_bar_B = evaluate_phi_bar_B(phi_A, phi_B, beta_A, k1)
    beta_bar_B_root = np.sqrt(beta_B) * np.sin(phi_B - phi_A) / np.sin(phi_bar_B - phi_A)
    return beta_bar_B_root**2

def evaluate_alpha_bar_B(alpha_B, beta_A, phi_A, phi_B, k1):
    phi_bar_B = evaluate_phi_bar_B(phi_A, phi_B, beta_A, k1)
    alpha_B_bar = 1/np.tan(phi_bar_B - phi_A) - \
                  (np.cos(phi_B - phi_A)*np.sin(phi_B - phi_A) - alpha_B * np.sin(phi_B - phi_A)**2) \
                    / np.sin(phi_bar_B - phi_A)**2
    return alpha_B_bar

def evaluate_matrix_after(alpha_A, alpha_B, beta_A, beta_B, phi_A, phi_B, k1):
    # Define S and C
    S = np.sin(phi_B - phi_A)
    C = np.cos(phi_B - phi_A)

    # Define the matrix elements
    element_11 = np.sqrt(beta_B / beta_A) * (C + alpha_A * S) - k1 * np.sqrt(beta_B * beta_A) * S
    element_12 = np.sqrt(beta_B * beta_A) * S
    element_21 = ((alpha_A - alpha_B) * C - (1 + alpha_A * alpha_B) * S) / np.sqrt(beta_B * beta_A) - k1 * np.sqrt(beta_A / beta_B) * (C - alpha_B * S)
    element_22 = np.sqrt(beta_A / beta_B) * (C - alpha_B * S)

    # Construct the matrix
    M = np.array([
        [element_11, element_12],
        [element_21, element_22]
    ])

    return M

mymat = evaluate_matrix(tw0.alfx[0], tw0.alfx[-1], tw0.betx[0], tw0.betx[-1],
                        2*np.pi*tw0.mux[0], 2*np.pi*tw0.mux[-1])
phi_formula = evaluate_phi_bar_B(2*np.pi*tw0.mux[0], 2*np.pi*tw0.mux[-1],
                                 tw0.betx[0], dk_test)
beta_formula = evaluate_beta_bar_B(tw0.betx[0], tw0.betx[-1],
                                   2*np.pi*tw0.mux[0], 2*np.pi*tw0.mux[-1], dk_test)
alpha_formula = evaluate_alpha_bar_B(tw0.alfx[-1], tw0.betx[0],
                                     2*np.pi*tw0.mux[0], 2*np.pi*tw0.mux[-1], dk_test)

env['dk1l'] = 1e-3
tw_after = line.twiss4d(init=tw0)
mat_after = evaluate_matrix_after(tw0.alfx[0], tw0.alfx[-1], tw0.betx[0], tw0.betx[-1],
                                    2*np.pi*tw0.mux[0], 2*np.pi*tw0.mux[-1], env['dk1l'])
tw_mat_after = tw_after.get_R_matrix(tw0.name[0], tw0.name[-1])[:2,:2]

import matplotlib.pyplot as plt
plt.close('all')

plt.figure(1)
plt.plot(dk_test*1e3, 2*np.pi*(mu_test - mu0), label=r'$\varphi_b$')
plt.plot(dk_test*1e3, dk_test*tw0.betx[0], '--', label=r'$\beta_a \Delta k$')
plt.plot(dk_test*1e3, phi_formula - 2*np.pi*mu0, 'x', label=r'$\bar{\varphi}_b$')
plt.legend()
plt.xlabel(r'$\Delta kL$')

plt.show()

plt.figure(1)
plt.plot(dk_test*1e3, alpha_test - alf0, label=r'$\alpha_B$')
#plt.plot(dk_test*1e3, dk_test*tw0.betx[0], '--', label=r'$\beta_a \Delta k$')
plt.plot(dk_test*1e3, alpha_formula - alf0, 'x', label=r'$\bar{\alpha}_b$')
plt.legend()
plt.xlabel(r'$\Delta kL$')

plt.show()

plt.figure(1)
plt.plot(dk_test*1e3, beta_test - bet0, label=r'$\beta_B$')
#plt.plot(dk_test*1e3, dk_test*tw0.betx[0], '--', label=r'$\beta_a \Delta k$')
plt.plot(dk_test*1e3, beta_formula - bet0, 'x', label=r'$\bar{\beta}_b$')
plt.legend()
plt.xlabel(r'$\Delta kL$')

plt.show()