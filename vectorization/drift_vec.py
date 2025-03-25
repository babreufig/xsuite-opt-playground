import xtrack as xt
import numpy as np
import time

n_part = int(1e6)
p = xt.Particles(p0c=7e12, x = np.linspace(-1, 1, n_part))

n_elem = 250
# generate random lengths between 0.1 and 5.0
lengths = np.random.rand(n_elem) * 4.9 + 0.1

line = xt.Line(elements=[xt.Drift(length=length) for length in lengths])

env = xt.Environment()
env.particle_ref = p

env['kq'] = 0.1

# line = env.new_line(components=[
#     env.new('qf', 'Quadrupole', k1='kq', length=1.0, anchor='start', at=0.),
#     env.new('qd', 'Quadrupole', k1='-kq', length=1.0, anchor='start', at=10.,
#             from_='qf@end'),
#     env.new('end', 'Marker', at=10., from_='qd@end')
# ])

#line = 10 * line

line.track(p)

n_exec = 4
elapsed_times = np.zeros(n_exec)

for i in range(n_exec):
    start_time = time.time()
    line.track(p)
    end_time = time.time()
    elapsed_times[i] = (end_time - start_time) * 1000  # Convert to milliseconds

elapsed_time = np.mean(elapsed_times)
time_std = np.std(elapsed_times)

print(f"Average time taken: {elapsed_time:.2f} ms\tStandard deviation: {time_std:.2f} ms")