from enum import IntEnum
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

# Start tracking code

class ElementType(IntEnum):
    DRIFT = 0
    QUADRUPOLE_KICK = 1

def drift(positions, init_cond, length):
    x, px, y, py, zeta, delta = positions
    rpp = 1 / (1 + delta)
    rvv, _ = init_cond
    return jnp.asarray([x + px * length,
                        px * rpp,
                        y + py * length,
                        py + rpp,
                        zeta + length * (1 - rvv * (1 + (px**2 + py**2)/2)),
                        delta])


def quad_kick(positions, init_cond, knl, ksl):
    x, px, y, py, zeta, delta = positions
    _, chi = init_cond
    return jnp.asarray([x,
                        px - chi * (knl * x + ksl * y),
                        y,
                        py + chi * (knl * y - ksl * x),
                        zeta,
                        delta])

def trackone(positions, init_cond, element):
    # Apply a drift or kick to the particle positions
    element_type, knl, ksl = element
    return jax.lax.cond(
        element_type == ElementType.DRIFT,
        lambda p: drift(p, init_cond, knl),
        lambda p: quad_kick(p, init_cond, knl, ksl),
        positions
    ), None


@jax.jit
def track(positions, init_cond, lattice_array):
    return jax.lax.scan(lambda p, e: trackone(p, init_cond, e), positions, lattice_array)[0]


dtrackfwd = jax.jit(jax.jacfwd(track))
dtrackrev = jax.jit(jax.jacrev(track))

@jax.jit
def dtrackfd(positions, init_cond, lattice_array, eps=1e-9):
    n_dim = len(positions)

    # Create a diagonal matrix for perturbation
    eye_eps = jnp.eye(n_dim) * eps  # Perturbation matrix

    # Forward and backward perturbed positions ( +eps and -eps)
    p_plus = positions + eye_eps  # Perturb the positions by +eps
    p_minus = positions - eye_eps  # Perturb the positions by -eps

    # Track the perturbed positions using the lattice
    tracked_plus = jax.vmap(lambda p: track(p, init_cond, lattice_array))(p_plus)
    tracked_minus = jax.vmap(lambda p: track(p, init_cond, lattice_array))(p_minus)

    # Compute the finite difference approximation to the Jacobian
    jacobian = (tracked_plus - tracked_minus) / (2 * eps)

    return jacobian.T

# End tracking code


ncell = 1
fodo_lattice = jnp.asarray([
    (ElementType.DRIFT, 1.2, 0.0),
    (ElementType.QUADRUPOLE_KICK, 0.8, 0.0),
    (ElementType.DRIFT, 1.2, 0.0),
    (ElementType.QUADRUPOLE_KICK, -0.7, -0.0),
] * ncell)

p0 = jnp.asarray([0.3, 0.1, 0.2, 0.4, 0.005, 0.001])

rvv = 1.1
chi = 0.8

init_cond = jnp.array([rvv, chi])

jac_fwd = dtrackfwd(p0, init_cond, fodo_lattice)
jac_rev = dtrackrev(p0, init_cond, fodo_lattice)
jac_fd = dtrackfd(p0, init_cond, fodo_lattice)

# tracked_pos = track(p0, init_cond, fodo_lattice)

# %timeit dtrackfwd(p0, init_cond, fodo_lattice)
# %timeit dtrackrev(p0, init_cond, fodo_lattice)
# %timeit dtrackfd(p0, init_cond, fodo_lattice)
