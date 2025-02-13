import xtrack as xt
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import time

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
    return 1/jnp.tan(x)

def arccot(x):
    # if x < 0:
    #     return jnp.arctan(1/x) + jnp.pi
    # return jnp.arctan(1/x)
    return jnp.pi / 2 - jnp.arctan(x) # Jittable because no condition

def evaluate_phi_bar_B(beta_A, phi_A, phi_B, k1):
    # Define the formula for phi_B_bar
    #phi_bar_B = np.arctan2(1, np.cos(phi_B - phi_A)/np.sin(phi_B - phi_A) - k1 * beta_A) + phi_A
    phi_bar_B = arccot(jnp.cos(phi_B - phi_A)/jnp.sin(phi_B - phi_A) - k1 * beta_A) + phi_A
    return phi_bar_B

def approximate_phi_bar_B(beta_A, phi_A, phi_B, k1):
    deriv_phi_bar_B_approx = k1 * beta_A / (1 + jnp.cos(phi_B - phi_A)**2/jnp.sin(phi_B - phi_A)**2) + phi_B
    return deriv_phi_bar_B_approx

def evaluate_alpha_bar_B(alpha_B, beta_A, phi_A, phi_B, k1):
    # Define the formula for alpha_B_bar
    phi_bar_B = evaluate_phi_bar_B(beta_A, phi_A, phi_B, k1)
    alpha_bar_B = cot(phi_bar_B - phi_A) - (jnp.cos(phi_B - phi_A)*jnp.sin(phi_B - phi_A) - alpha_B * jnp.sin(phi_B - phi_A)**2)\
                                            /jnp.sin(phi_bar_B - phi_A)**2
    return alpha_bar_B

def approximate_alpha_bar_B(alpha_B, beta_A, phi_A, phi_B, k1):
    alpha_bar_B_approx = alpha_B + k1 * beta_A * \
                (jnp.cos(phi_B - phi_A)**2 -
                 jnp.sin(phi_B - phi_A)**2 -
                 2 * alpha_B * jnp.cos(phi_B - phi_A) * jnp.sin(phi_B - phi_A))
    return alpha_bar_B_approx

def evaluate_beta_bar_B(beta_A, beta_B, phi_A, phi_B, k1):
    # Define the formula for beta_B_bar
    phi_bar_B = evaluate_phi_bar_B(beta_A, phi_A, phi_B, k1)
    beta_bar_B_root = jnp.sqrt(beta_B) * jnp.sin(phi_B - phi_A) / jnp.sin(phi_bar_B - phi_A)
    return beta_bar_B_root**2

def approximate_beta_bar_B(beta_A, beta_B, phi_A, phi_B, k1):
    beta_bar_B_approx = beta_B - 2 * beta_B * k1 * beta_A * jnp.sin(phi_B - phi_A) * jnp.cos(phi_B - phi_A)
    return beta_bar_B_approx

def deriv_phi_b_bar(beta_A, phi_A, phi_B, k1l):
    # Define the derivative of phi_B_bar with respect to k1l
    dphi = beta_A / (1 + (cot(phi_B - phi_A) - k1l * beta_A)**2)
    return dphi

def approximate_deriv_phi_b_bar(beta_A, phi_A, phi_B):
    deriv_phi_b_bar = beta_A / (1 + jnp.cos(phi_B - phi_A)**2/jnp.sin(phi_B - phi_A)**2)
    return deriv_phi_b_bar

def deriv_beta_b_bar(beta_A, beta_B, phi_A, phi_B, k1l):
    # Define the derivative of beta_B_bar with respect to k1l
    phi_B_bar = evaluate_phi_bar_B(beta_A, phi_A, phi_B, k1l)
    dbeta = beta_B * np.sin(phi_B - phi_A)**2 * (-2 * np.cos(phi_B_bar - phi_A)) / np.sin(phi_B_bar - phi_A)**3 * deriv_phi_b_bar(beta_A, phi_A, phi_B, k1l)
    #dbeta = -1 * beta_A * beta_B * np.sin(phi_B - phi_A)**2 * (cot(phi_B - phi_A) - k1l * beta_A)
    return dbeta

def approximate_deriv_beta_b_bar(beta_A, beta_B, phi_A, phi_B):
    deriv_beta_b_bar_approx = -2 * beta_B * beta_A * jnp.sin(phi_B - phi_A) * jnp.cos(phi_B - phi_A)
    return deriv_beta_b_bar_approx

def deriv_alpha_b_bar(alpha_B, beta_A, phi_A, phi_B, k1l):
    # Define the derivative of alpha_B_bar with respect to k1l
    dalpha = 2 * beta_A * (cot(phi_B - phi_A) - k1l * beta_A) * \
                (np.cos(phi_B - phi_A) * np.sin(phi_B - phi_A) - alpha_B * np.sin(phi_B - phi_A)**2) - beta_A
    return dalpha

def approximate_deriv_alpha_b_bar(alpha_B, beta_A, phi_A, phi_B):
    deriv_alpha_b_bar_approx = beta_A * \
                (jnp.cos(phi_B - phi_A)**2 -
                 jnp.sin(phi_B - phi_A)**2 -
                 2 * alpha_B * jnp.cos(phi_B - phi_A) * jnp.sin(phi_B - phi_A))
    return deriv_alpha_b_bar_approx


tw0 = line.twiss4d()

alpha_a = tw0.alfx[0]
alpha_b = tw0.alfx[-1]
beta_a = tw0.betx[0]
beta_b = tw0.betx[-1]
phi_a = 2*np.pi*tw0.mux[0]
phi_b = 2*np.pi*tw0.mux[-1]

dk1l = env['dk1l']
step = 1e-8
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


start = time.perf_counter()
dbetabform = deriv_beta_b_bar(beta_a, beta_b, phi_a, phi_b, dk1l)
dphibform = deriv_phi_b_bar(beta_a, phi_a, phi_b, dk1l)
dalphabform = deriv_alpha_b_bar(alpha_b, beta_a, phi_a, phi_b, dk1l)
end = time.perf_counter()

print(f"Calculating derivatives analytically took: {(end-start)*1e3:0.4f} milliseconds")

start = time.perf_counter()
dbetabform = approximate_deriv_beta_b_bar(beta_a, beta_b, phi_a, phi_b)
dphibform = approximate_deriv_phi_b_bar(beta_a, phi_a, phi_b)
dalphabform = approximate_deriv_alpha_b_bar(alpha_b, beta_a, phi_a, phi_b)
end = time.perf_counter()

print(f"Calculating derivatives approximatively took: {(end-start)*1e3:0.4f} milliseconds")

num_elem = 1000
x = np.linspace(-0.001, 0.001, num_elem)

start_alpha = time.perf_counter()
dalpha = [deriv_alpha_b_bar(alpha_b, beta_a, phi_a, phi_b, xi) for xi in x]
end_alpha = time.perf_counter()
print(f"Calculating alpha derivative analytically (list of {num_elem} elem) took: {(end_alpha-start_alpha)*1e3:0.4f} milliseconds")
start_beta = time.perf_counter()
dbeta = [deriv_beta_b_bar(beta_a, beta_b, phi_a, phi_b, xi) for xi in x]
end_beta = time.perf_counter()
print(f"Calculating beta derivative analytically (list of {num_elem} elem) took: {(end_beta-start_beta)*1e3:0.4f} milliseconds")
start_phi = time.perf_counter()
dphi = [deriv_phi_b_bar(beta_a, phi_a, phi_b, xi) for xi in x]
end_phi = time.perf_counter()
print(f"Calculating phi derivative analytically (list of {num_elem} elem) took: {(end_phi-start_phi)*1e3:0.4f} milliseconds")

deriv_analytical_list = end_phi - start_alpha
print(f"Calculating derivative for all (list of {num_elem}) analytically took: {deriv_analytical_list*1e3:0.4f} milliseconds")

start_alpha = time.perf_counter()
dalpha = [approximate_deriv_alpha_b_bar(alpha_b, beta_a, phi_a, phi_b) for xi in x]
end_alpha = time.perf_counter()
print(f"Calculating alpha derivative approximatively (list of {num_elem} elem) took: {(end_alpha-start_alpha)*1e3:0.4f} milliseconds")
start_beta = time.perf_counter()
dbeta = [approximate_deriv_beta_b_bar(beta_a, beta_b, phi_a, phi_b) for xi in x]
end_beta = time.perf_counter()
print(f"Calculating beta derivative approximatively (list of {num_elem} elem) took: {(end_beta-start_beta)*1e3:0.4f} milliseconds")
start_phi = time.perf_counter()
dphi = [approximate_deriv_phi_b_bar(beta_a, phi_a, phi_b) for xi in x]
end_phi = time.perf_counter()
print(f"Calculating phi derivative approximatively (list of {num_elem} elem) took: {(end_phi-start_phi)*1e3:0.4f} milliseconds")

deriv_analytical_list = end_phi - start_alpha
print(f"Calculating derivative for all (list of {num_elem}) approximatively took: {deriv_analytical_list*1e3:0.4f} milliseconds.\n"+\
      "This can be optimized, since it's a constant function and just needs to be calculated once.")


# ----------------------------------
# JAX part

import jax.numpy as jnp
import jax

start = time.perf_counter()
grad_beta_bar = jax.grad(evaluate_beta_bar_B, argnums=4)
grad_alpha_bar = jax.grad(evaluate_alpha_bar_B, argnums=4)
grad_phi_bar = jax.grad(evaluate_phi_bar_B, argnums=3)
end = time.perf_counter()

print(f"Creating gradient function took: {(end-start)*1e3:0.4f} milliseconds")

start = time.perf_counter()
grad_beta_bar_approx = jax.grad(approximate_beta_bar_B, argnums=4)
grad_alpha_bar_approx = jax.grad(approximate_alpha_bar_B, argnums=4)
grad_phi_bar_approx = jax.grad(approximate_phi_bar_B, argnums=3)
end = time.perf_counter()

print(f"Creating gradient function took: {(end-start)*1e3:0.4f} milliseconds")

start_alpha = time.perf_counter()
dalpha_autodiff = grad_alpha_bar(alpha_b, beta_a, phi_a, phi_b, dk1l)
end_alpha = time.perf_counter()
print(f"Calculating alpha derivative (1st) took: {(end_alpha-start_alpha)*1e3:0.4f} milliseconds")
start_beta = time.perf_counter()
dbeta_autodiff = grad_beta_bar(beta_a, beta_b, phi_a, phi_b, dk1l)
end_beta = time.perf_counter()
print(f"Calculating beta derivative (1st) took: {(end_beta-start_beta)*1e3:0.4f} milliseconds")
start_phi = time.perf_counter()
dphi_autodiff = grad_phi_bar(beta_a, phi_a, phi_b, dk1l)
end_phi = time.perf_counter()
print(f"Calculating phi derivative (1st) took: {(end_phi-start_phi)*1e3:0.4f} milliseconds")

deriv_first_time = end_phi - start_alpha
print(f"Calculating derivative for all (1st) took: {deriv_first_time*1e3:0.4f} milliseconds")

start_alpha = time.perf_counter()
dalpha_autodiff_list = [grad_alpha_bar(alpha_b, beta_a, phi_a, phi_b, xi) for xi in x]
end_alpha = time.perf_counter()
print(f"Calculating alpha derivative (list of {num_elem} elem) took: {(end_alpha-start_alpha)*1e3:0.4f} milliseconds")
start_beta = time.perf_counter()
dbeta_autodiff_list = [grad_beta_bar(beta_a, beta_b, phi_a, phi_b, xi) for xi in x]
end_beta = time.perf_counter()
print(f"Calculating beta derivative (list of {num_elem} elem) took: {(end_beta-start_beta)*1e3:0.4f} milliseconds")
start_phi = time.perf_counter()
dphi_autodiff_list = [grad_phi_bar(beta_a, phi_a, phi_b, xi) for xi in x]
end_phi = time.perf_counter()
print(f"Calculating phi derivative (list of {num_elem} elem) took: {(end_phi-start_phi)*1e3:0.4f} milliseconds")

deriv_list = end_phi - start_alpha
print(f"Calculating derivative for all (list of {num_elem} elem) took: {deriv_list*1e3:0.4f} milliseconds")

# start_alpha = time.perf_counter()
grad_alpha_jit = jax.jit(grad_alpha_bar)
# end_alpha = time.perf_counter()
# print(f"Jitting alpha derivative took: {(end_alpha-start_alpha)*1e3:0.4f} milliseconds")
# start_beta = time.perf_counter()
grad_beta_jit = jax.jit(grad_beta_bar)
# end_beta = time.perf_counter()
# print(f"Jitting beta derivative took: {(end_beta-start_beta)*1e3:0.4f} milliseconds")
# start_phi = time.perf_counter()
grad_phi_jit = jax.jit(grad_phi_bar)
# end_phi = time.perf_counter()
# print(f"Jitting phi derivative took: {(end_phi-start_phi)*1e3:0.4f} milliseconds")

# jitting_time = end_phi - start_alpha
# print(f"Jitting all derivatives took: {jitting_time*1e3:0.4f} milliseconds")

grad_alpha_approx_jit = jax.jit(grad_alpha_bar_approx)
grad_beta_approx_jit = jax.jit(grad_beta_bar_approx)
grad_phi_approx_jit = jax.jit(grad_phi_bar_approx)

start_alpha = time.perf_counter()
dalpha_autodiff_jit = grad_alpha_jit(alpha_b, beta_a, phi_a, phi_b, dk1l).block_until_ready()
end_alpha = time.perf_counter()
print(f"Calculating alpha derivative (1st) with jit took: {(end_alpha-start_alpha)*1e3:0.4f} milliseconds")
start_beta = time.perf_counter()
dbeta_autodiff_jit = grad_beta_jit(beta_a, beta_b, phi_a, phi_b, dk1l).block_until_ready()
end_beta = time.perf_counter()
print(f"Calculating beta derivative (1st) with jit took: {(end_beta-start_beta)*1e3:0.4f} milliseconds")
start_phi = time.perf_counter()
dphi_autodiff_jit = grad_phi_jit(beta_a, phi_a, phi_b, dk1l).block_until_ready()
end_phi = time.perf_counter()
print(f"Calculating phi derivative (1st) with jit took: {(end_phi-start_phi)*1e3:0.4f} milliseconds")

deriv_jit_time = end_phi - start_alpha
print(f"Calculating derivative for all (1st) with jit took: {deriv_jit_time*1e3:0.4f} milliseconds")

start_alpha = time.perf_counter()
dalpha_autodiff_jit = grad_alpha_approx_jit(alpha_b, beta_a, phi_a, phi_b, dk1l).block_until_ready()
end_alpha = time.perf_counter()
print(f"Calculating approximate alpha derivative (1st) with jit took: {(end_alpha-start_alpha)*1e3:0.4f} milliseconds")
start_beta = time.perf_counter()
dbeta_autodiff_jit = grad_beta_approx_jit(beta_a, beta_b, phi_a, phi_b, dk1l).block_until_ready()
end_beta = time.perf_counter()
print(f"Calculating approximate beta derivative (1st) with jit took: {(end_beta-start_beta)*1e3:0.4f} milliseconds")
start_phi = time.perf_counter()
dphi_autodiff_jit = grad_phi_approx_jit(beta_a, phi_a, phi_b, dk1l).block_until_ready()
end_phi = time.perf_counter()
print(f"Calculating approximate phi derivative (1st) with jit took: {(end_phi-start_phi)*1e3:0.4f} milliseconds")

deriv_jit_time = end_phi - start_alpha
print(f"Calculating approximate derivative for all (1st) with jit took: {deriv_jit_time*1e3:0.4f} milliseconds")

start_alpha = time.perf_counter()
dalpha_autodiff_jit = [grad_alpha_jit(alpha_b, beta_a, phi_a, phi_b, xi).block_until_ready() for xi in x]
end_alpha = time.perf_counter()
print(f"Calculating alpha derivative (list with {num_elem} elem) with jit took: {(end_alpha-start_alpha)*1e3:0.4f} milliseconds")
start_beta = time.perf_counter()
dbeta_autodiff_jit = [grad_beta_jit(beta_a, beta_b, phi_a, phi_b, xi).block_until_ready() for xi in x]
end_beta = time.perf_counter()
print(f"Calculating beta derivative (list with {num_elem} elem) with jit took: {(end_beta-start_beta)*1e3:0.4f} milliseconds")
start_phi = time.perf_counter()
dphi_autodiff_jit = [grad_phi_jit(beta_a, phi_a, phi_b, xi).block_until_ready() for xi in x]
end_phi = time.perf_counter()
print(f"Calculating phi derivative (list with {num_elem} elem) with jit took: {(end_phi-start_phi)*1e3:0.4f} milliseconds")

deriv_jit_time_list = end_phi - start_alpha
print(f"Calculating derivative for all (list with {num_elem}) with jit took: {deriv_jit_time_list*1e3:0.4f} milliseconds")

start_alpha = time.perf_counter()
dalpha_autodiff_jit = [grad_alpha_approx_jit(alpha_b, beta_a, phi_a, phi_b, xi).block_until_ready() for xi in x]
end_alpha = time.perf_counter()
print(f"Calculating approximate alpha derivative (list with {num_elem} elem) with jit took: {(end_alpha-start_alpha)*1e3:0.4f} milliseconds")
start_beta = time.perf_counter()
dbeta_autodiff_jit = [grad_beta_approx_jit(beta_a, beta_b, phi_a, phi_b, xi).block_until_ready() for xi in x]
end_beta = time.perf_counter()
print(f"Calculating approximate beta derivative (list with {num_elem} elem) with jit took: {(end_beta-start_beta)*1e3:0.4f} milliseconds")
start_phi = time.perf_counter()
dphi_autodiff_jit = [grad_phi_approx_jit(beta_a, phi_a, phi_b, xi).block_until_ready() for xi in x]
end_phi = time.perf_counter()
print(f"Calculating approximate phi derivative (list with {num_elem} elem) with jit took: {(end_phi-start_phi)*1e3:0.4f} milliseconds")

deriv_jit_time_list = end_phi - start_alpha
print(f"Calculating approximate derivative for all (list with {num_elem}) with jit took: {deriv_jit_time_list*1e3:0.4f} milliseconds")