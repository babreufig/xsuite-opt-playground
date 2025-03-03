# This code is based on an example using MAiNGO, accessible under
# https://git.rwth-aachen.de/avt-svt/public/maingo/-/blob/master/examples/01_BasicExample/examplePythonInterface.py
# MAiNGO is licensed under the Eclipse Public License v2.0 (EPL 2.0)

from maingopy import *
import xtrack as xt
import os
import numpy as np

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

#####################################################
# Define a model
class Model(MAiNGOmodel):
    def __init__(self):
        MAiNGOmodel.__init__(self)


    # We need to implement the get_variables functions for specifying the optimization variables
    def get_variables(self):
        # We need to return a list of OptimizationVariable objects.
        # To define an optimization variable, we typically need to specify bounds, and optionally a variable type, a branching priority, and a name
        #
        # Variable bounds:
        #  Every optimization variable (except for binary variables, cf. below) requires finite lower and upper bounds.
        #
        # Variable type:
        #  There are three variable types in MAiNGO. VT_CONTINUOUS variables, VT_BINARY variables, and VT_INTEGER variables.
        #  Double type bounds for binaries and integers will be rounded up for lower bounds and rounded down for upper bounds.
        #
        # Branching priority:
        #  A branching priority 'n' means that we will branch log_2(n+1) times more often on that specific variable, n will be rounded down to the next integer meaning that a BP of 1.5 equals 1
        #  If you want to branch less on a specific variable just increase the branching priority of all other variables
        #  A branching priority of <1 means that MAiNGO will never branch on this specific variable. This may lead to non-convergence of the B&B algorithm
        #
        # Variable name:
        #  The name has to be of string type. All ASCII characters are allowed and variables are allowed to have the same name.
        #  MAiNGO outputs the variables in the same order as they are set in the variables list within this function.
        #

        variables = [OptimizationVariable(Bounds(-1e-4, 1e-4), VT_CONTINUOUS, "kqtf.b1"),
                     OptimizationVariable(Bounds(-1e-4, 1e-4), VT_CONTINUOUS, "kqtd.b1"),
                     OptimizationVariable(Bounds(-0.1, 0.1), VT_CONTINUOUS, "ksf.b1"),
                     OptimizationVariable(Bounds(-0.1, 0.1), VT_CONTINUOUS, "ksd.b1")]
        return variables

    # Optional: we can implement a function for specifying an initial guess for our optimization variables
    # If provided, MAiNGO will use this point for the first local search during pre-processing
    def get_initial_point(self):
        # If you choose to provide an initial point, you have to make sure that the size of the initialPoint equals the size of
        # the variables list returned by get_variables. Otherwise, MAiNGO will throw an exception.
        # The value of an initial point variable does not have to fit the type of the variable, e.g., it is allowed to set a double type value as an initial point for a binary variable
        initialPoint = merit_function.get_x()
        return initialPoint


    # We need to implement the evaluate function that computes the values of the objective and constraints from the variables.
    # Note that the variables in the 'vars' argument of this function do correspond to the optimization variables defined in the get_variables function.
    # However, they are different objects for technical reasons. The only mapping we have between them is the position in the list.
    # The results of the evaluation (i.e., objective and constraint values) need to be returned in an EvaluationContainer
    def evaluate(self, vars):
        # The objective and constraints are returned in an EvaluationContainer
        result = EvaluationContainer()

        # Example objective: the Ackley function

        result.objective = merit_function(vars) # Crashes here, NumPy does not support FFVars

        #res = merit_function(vars)

        result.output = [OutputVariable("qx: ", result.objective[0]), OutputVariable("qy: ", result.objective[1]),
                         OutputVariable("dqx: ", result.objective[2]), OutputVariable("dqy: ", result.objective[3])]

        return result


#####################################################
# To work with the problem, we first create an instance of the model.
myModel = Model()

# We then create an instance of MAiNGO, the solver, and hand it the model.
myMAiNGO = MAiNGO(myModel)

# Next, adjust settings as desired
# We can have MAiNGO read a settings file:
fileName = ""
myMAiNGO.read_settings(fileName) # If fileName is empty, MAiNGO will attempt to open MAiNGOSettings.txt

#only for the parallel version
if HAVE_MAiNGO_MPI():
    #wokers must be unmuted before solving model
    unmuteWorker(buffer)

# Finally, we call the solve routine to solve the problem.
maingoStatus = myMAiNGO.solve()
