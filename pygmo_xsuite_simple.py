import xtrack as xt
import os
import numpy as np
import pygmo as pg

dir_path = os.path.dirname(os.path.realpath(__file__))

# Load a line and build a tracker
line = xt.Line.from_json(dir_path + '/lhc_thick_with_knobs.json')
line.build_tracker()

# Match tunes and chromaticities to assigned values
opt = line.match(
    solve=False,
    method='4d',
    vary=[
        xt.VaryList(['kqtf.b1', 'kqtd.b1'], step=1e-8, tag='quad'),
        xt.VaryList(['ksf.b1', 'ksd.b1'], step=1e-4, limits=[-0.1, 0.1], tag='sext'),
    ],
    targets = [
        xt.TargetSet(qx=62.315, qy=60.325, tol=1e-6, tag='tune'),
        xt.TargetSet(dqx=10.0, dqy=12.0, tol=0.01, tag='chrom'),
    ])

merit_function = opt.get_merit_function(return_scalar=False, check_limits=False)

class Xsuite_simple:

    def fitness(self, x):
        y = merit_function(x)
        y[0:2] *= 10 # Stronger weight on the first two targets (weight = 10)
        y = y**2 # Avoid negative errors
        return y
    
    def get_nic(self):
        return 0 # Number of inequality constraints
    
    def get_nec(self):
        return 0 # Number of equality constraints
    
    def get_nx(self):
        return 4
    
    def get_bounds(self):
        return ([[-1e-4, -1e-4, -0.1, -0.1], [1e-4, 1e-4, 0.1, 0.1]])
    
    def get_nobj(self):
        return 4 # Number of objectives
    
    def gradient(self, x):
        pg.estimate_gradient_h(lambda x: self.fitness(x), x)

prob = pg.problem(Xsuite_simple())

# create population
pop = pg.population(prob, size=80)

# select algorithm
algo = pg.algorithm(pg.nsga2(gen=50))
# run optimization
pop = algo.evolve(pop)
# extract results
fits, vectors = pop.get_f(), pop.get_x()
# extract and print non-dominated fronts
ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(fits)

sumfits = np.sum(fits, axis=1)
argmin = np.argmin(sumfits)

print(fits[argmin], vectors[argmin])

merit_function.set_x(vectors[argmin])
opt.target_status()