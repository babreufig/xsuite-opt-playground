from enum import IntEnum
import numpy as np
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
        lambda p: quad_kick(p, init_cond, knl, ksl), # knl = param, ksl = 0.0
        positions
    ), None


@jax.jit
def track(positions, init_cond, lattice_array):
    return jax.lax.scan(lambda p, e: trackone(p, init_cond, e), positions, lattice_array)[0]


dtrackfwd = jax.jit(jax.jacfwd(track))
dtrackrev = jax.jit(jax.jacrev(track))


@jax.jit
def dtrackfd(positions, init_cond, lst, eps=1e-12):
    eps_diag = jnp.eye(len(positions)) * eps
    positions_vector = jnp.vstack([positions]*len(positions)).T
    positions_vector = track(jnp.vstack(
        [positions_vector+eps_diag, positions_vector-eps_diag]).T, init_cond, lst)
    return (positions_vector[:, :len(positions)]-positions_vector[:, len(positions):])/(2*eps)

# End tracking code


ncell = 10000
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

# jac_fwd = dtrackfwd(p0, init_cond, fodo_lattice)
# jac_rev = dtrackrev(p0, init_cond, fodo_lattice)
# jac_fd = dtrackfd(p0, init_cond, fodo_lattice)

# tracked_pos = track(p0, init_cond, fodo_lattice)

# %timeit dtrackfwd(p0, init_cond, fodo_lattice)
# %timeit dtrackrev(p0, init_cond, fodo_lattice)
# %timeit dtrackfd(p0, init_cond, fodo_lattice)
