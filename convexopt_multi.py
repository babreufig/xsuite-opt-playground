from cvxopt import matrix, solvers

# Multivariate example for cvxopt.cp:
# Minimize functions f0(x,y) = (x-1)² + (y-2)² ; f1(x,y) = (x + y - 3)² ; f2(x, y) = (x - y)² (two variables, one target)
# under constraints:
# 1. x + y >= 3
# 2. x >= 0
# 3. y >= 0
# 4. x - y = 1

def F(x=None, z=None):
    if x is None:  # Initial guess (2.2, 1.2), zero nonlinear functions
        return 0, matrix([2.2, 1.2])
    
    # Zielfunktion
    f = matrix([(x[0] - 1)**2 + (x[1] - 2)**2 +  # f1(x, y)
                (x[0] + x[1] - 3)**2 +            # f2(x, y)
                (x[0] - x[1])**2],               # f3(x, y)
               (1, 1))
    
    # Gradient
    Df = matrix([2 * (x[0] - 1) + 2 * (x[0] + x[1] - 3) + 2 * (x[0] - x[1]),  # Gradient of f1
                 2 * (x[1] - 2) + 2 * (x[0] + x[1] - 3) + -2 * (x[0] - x[1]),  # Gradient of f2
                 ]).T  # Gradient of f3
    
    if z is None:
        return f, Df
    
    H = z[0] * (matrix([[2.0, 0.0], [0.0, 2.0]]) + matrix([[2.0, 2.0], [2.0, 2.0]]) + matrix([[2.0, -2.0], [-2.0, 2.0]]))
    return f, Df, H

# ## Does not work. For a vector function, needs to sum all functions, as well as Jacobians and Hessians
# def F_multi(x=None, z=None):
#     if x is None:  # Initial guess (5, 4), zero nonlinear functions
#         return 2, matrix([2.0, 1.0])
    
#     # Zielfunktion
#     f = matrix([(x[0] - 1)**2 + (x[1] - 2)**2,  # f1(x, y)
#                 (x[0] + x[1] - 3)**2,            # f2(x, y)
#                 (x[0] - x[1])**2],               # f3(x, y)
#                (3, 1))
    
#     # Gradient
#     Df = matrix([[2 * (x[0] - 1), 2 * (x[1] - 2)],  # Gradient of f1
#                  [2 * (x[0] + x[1] - 3), 2 * (x[0] + x[1] - 3)],  # Gradient of f2
#                  [2 * (x[0] - x[1]), -2 * (x[0] - x[1])]]).T  # Gradient of f3
    
#     if z is None:
#         return f, Df
    
#     H = z[0] * matrix([[2.0, 0.0], [0.0, 2.0]]) + \
#         z[1] * matrix([[2.0, 2.0], [2.0, 2.0]]) + \
#         z[2] * matrix([[2.0, -2.0], [-2.0, 2.0]])
#     return f, Df, H

# Inequality Constraints:
G = matrix([[-1.0, -1.0], [-1.0, 0.0], [0.0, -1.0]]).T
h = matrix([3.0, 0.0, 0.0])

## Equality Constraints

A = matrix([1.0, -1.0]).T
b = matrix([1.0])

# Solve
solution = solvers.cp(F, G=G, h=h, A=A, b=b)

# Optimal solution
x_opt = solution['x']
optimal_value = solution['primal objective']

print("Optimal solution (x):", x_opt[0], x_opt[1])
print("Minimal value of the function:", optimal_value)