import xtrack as xt
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

sp.init_printing()

env = xt.Environment()
env.particle_ref = xt.Particles(p0c=7e12)

env['kq'] = 0.1
env['dk1l'] = 0.
env['bphi'] = 0.

line = env.new_line(components=[
    env.new('qf', 'Quadrupole', k1='kq', length=0.5, anchor='start', at=1.),
    env.new('bend', 'Bend', angle='bphi', k0='bphi', at=0., anchor='start', length=1.0),
    env.new('qd', 'Quadrupole', k1='-kq', length=0.5, anchor='start', at=11.,
            from_='qf@end'),
    env.new('end', 'Marker', at=10., from_='qd@end'),
    env.new('start', 'Marker', at=0.),
    env.new('dquad', 'Multipole', knl=[0., 'dk1l'], at=1.),
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

#tw0 = line.twiss4d(start=xt.START, end="qd@end", init_at=xt.END)
tw0 = line.twiss4d()
init_conditions = tw0.get_twiss_init('start')

env['bphi'] = 2 * np.pi / 1000
tw0 = line.twiss4d(init=init_conditions)

betx_A = tw0.betx[0]
bety_A = tw0.bety[0]
betx_B = tw0.betx[-1]
bety_B = tw0.bety[-1]
alfx_A = tw0.alfx[0]
alfy_A = tw0.alfy[0]
alfx_B = tw0.alfx[-1]
alfy_B = tw0.alfy[-1]
mux_A = tw0.mux[0]
muy_A = tw0.muy[0]
mux_B = tw0.mux[-1]
muy_B = tw0.muy[-1]
dx_A = tw0.dx[0]
dx_B = tw0.dx[-1]
dy_A = tw0.dy[0]
dy_B = tw0.dy[-1]
dpx_A = tw0.dpx[0]
dpx_B = tw0.dpx[-1]
dpy_A = tw0.dpy[0]
dpy_B = tw0.dpy[-1]

R_AB = tw0.get_R_matrix(tw0.name[0], tw0.name[-1])

print(dx_A, dpx_A, dy_A, dpy_A)
print(dx_B, dpx_B, dy_B, dpy_B)

dk1l_arr = np.linspace(-0.001, 0.001, 100)
init_sec = tw0.get_twiss_init('start')
dx_arr = np.zeros(len(dk1l_arr))

for i, dk1l in enumerate(dk1l_arr):
    env['dk1l'] = dk1l
    twnew = line.twiss(init=init_sec)
    dx_arr[i] = twnew.dx[-1]
    analytic_ddx = twnew.get_twiss_param_derivative('end', 'dquad')['ddx']

gradient = (dx_arr[1:] - dx_arr[:-1])/(dk1l_arr[1] - dk1l_arr[0])

print(analytic_ddx)
print(gradient)

plt.show()