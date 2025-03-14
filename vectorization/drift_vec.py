import xtrack as xt
import numpy as np
import time

n_part = int(5e7)
p = xt.Particles(p0c=7e12, x = np.linspace(-1, 1, n_part))

line = xt.Line(elements=[xt.Drift(length=1.0)])

start_time = time.time()
line.track(p)
end_time = time.time()

elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
print(f"Time taken: {elapsed_time:.2f} ms")