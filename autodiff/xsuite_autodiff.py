import xtrack as xt
import numpy as np
import matplotlib.pyplot as plt

env = xt.Environment()
env.particle_ref = xt.Particles(p0c=7e12)

env['kq'] = 0.1

env.new('qf', 'Quadrupole', k1='kq', length=1.0, anchor='start')
env.new('qd', 'Quadrupole', k1='-kq', length=1.0, anchor='start')
env.new('drift', 'Drift', length=1.0)
env.new('end', 'Marker', at=10., from_='qd@end')

line = env.new_line(components=[
    env.place('qf', anchor='start', at=0.),
    env.place('drift', at=1., from_='qf@end'),
    env.place('qd', anchor='start', at=10., from_='qf@end'),
    env.place('drift', anchor='start', at=11., from_='qd@end'),
    env.place('end', at=10., from_='drift@end'),
])

opt = line.match(
    method='4d',
    solve=False,
    vary=xt.Vary('kq', step=1e-4),
    targets=xt.Target('qx', 0.166666, tol=1e-6),
)
opt.solve()

n_cells = 10000
line_full = n_cells * line

line_full.build_tracker()

n_part = 1
particles = line_full.build_particles(x=0.3, px=0.1, y=0.2, py=0.4)

%timeit line_full.track(particles)

print(line.record_last_track)