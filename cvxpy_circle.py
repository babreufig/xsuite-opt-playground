import cvxpy as cp
import numpy as np

x = cp.Variable()
y = cp.Variable()

objective = cp.Minimize(x)

constraint1 = x**2 + y**2 <= 4 # convex
constraint2 = x**2 + y**2 >= 3 # concave

objective_with_penalty = cp.Minimize(x)

problem = cp.Problem(objective_with_penalty, [constraint1, constraint2])

# Throws error because constraint2 is not convex
# thus doesn't follow the rules of disciplined convex programming
problem.solve()

print("Optimal value of x:", x.value)
print("Optimal value of y:", y.value)
print("Minimum value of f(x, y):", problem.value)