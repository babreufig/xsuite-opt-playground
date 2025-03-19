import xtrack as xt
import numpy as np

env = xt.Environment()
env.particle_ref = xt.Particles(p0c=7e12)

env['kq'] = 0.1
env['dk1l'] = 0.
env['bphi'] = 0.01

src_marker = 'src_mk'
src_marker_loc = 3.
obs_marker = 'obs_mk'

line = env.new_line(components=[
    env.new('qf', 'Quadrupole', k1='kq', length=0.5, anchor='start', at=1.),
    env.new('qd', 'Quadrupole', k1='-kq', length=0.5, anchor='start', at=11.,
            from_='qf@end'),
    env.new('bendh', 'Bend', angle='bphi', k0_from_h=True, at=5., anchor='start', length=1.0),
    env.new('bendv', 'Bend', angle='bphi', rot_s_rad=np.pi/2, k0_from_h=True,
            at=18., anchor='start', length=1.0),
    env.new('end', 'Marker', at=10., from_='qd@end'),
    env.new('start', 'Marker', at=0.),
    env.new(src_marker, 'Marker', at=src_marker_loc),
    env.new(obs_marker, 'Marker', at=15.),
    env.new('dquad', 'Multipole', knl=[0., 'dk1l'], at=src_marker_loc),
])

# opt = line.match(
#     method='4d',
#     solve=False,
#     vary=xt.Vary('kq', step=1e-4),
#     targets=xt.Target('qx', 0.166666, tol=1e-6),
# )

# opt.solve()
# opt.target_status()
# opt.vary_status()

tw0 = line.twiss4d()

tw1 = line.twiss4d(start=xt.START, end=src_marker, init=tw0)
tw2 = line.twiss4d(start=src_marker, end=obs_marker, init=tw0)
tw3 = line.twiss4d(start=obs_marker, end=xt.END, init=tw0)

tw_deriv = tw2.get_twiss_param_derivative(src=src_marker, observation=obs_marker)

R_AB = tw0.get_R_matrix(src_marker, obs_marker)

eps = 1e-4

env['dk1l'] += eps
tw_plus = line.twiss4d(init=tw0, start=src_marker, end=obs_marker)
env['dk1l'] -= 2 * eps
tw_minus = line.twiss4d(init=tw0, start=src_marker, end=obs_marker)
env['dk1l'] += eps

fd_dict = {
    'betx': (tw_plus.betx[-1] - tw_minus.betx[-1]) / (2 * eps),
    'bety': (tw_plus.bety[-1] - tw_minus.bety[-1]) / (2 * eps),
    'alfx': (tw_plus.alfx[-1] - tw_minus.alfx[-1]) / (2 * eps),
    'alfy': (tw_plus.alfy[-1] - tw_minus.alfy[-1]) / (2 * eps),
    'mux': (tw_plus.mux[-1] - tw_minus.mux[-1]) / (2 * eps),
    'muy': (tw_plus.muy[-1] - tw_minus.muy[-1]) / (2 * eps),
    'dx': (tw_plus.dx[-1] - tw_minus.dx[-1]) / (2 * eps),
    'dpx': (tw_plus.dpx[-1] - tw_minus.dpx[-1]) / (2 * eps),
    'dy': (tw_plus.dy[-1] - tw_minus.dy[-1]) / (2 * eps),
    'dpy': (tw_plus.dpy[-1] - tw_minus.dpy[-1]) / (2 * eps),
}

# pretty print both fd_dict and tw_deriv (use pretty print)
import pprint
pprint.pprint("Finite Differences:")
pprint.pprint(fd_dict)
pprint.pprint(f"Twiss Derivatives:")
pprint.pprint(tw_deriv)