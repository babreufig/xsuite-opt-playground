from cvxopt import matrix, solvers
import numpy as np

# Circle Example for cvxopt.cp:
# Minimize function f(x,y) = x (two variables, one target)
# under constraints:
# 1. x² + y² >= 3 ---->
# 2. x² + y² <= 4

def f1(x, y):
    return x**2 + y**2 # convex

def f_1(x):
    return -x[0]**2 - x[1]**2 + 3.99 # concave

def f_2(x):
    return 2*(x[0]**2 + x[1]**2 - 4) # convex

starting_point = [-0.5, -1.2]

def F(x=None, z=None):
    if x is None:  # Initial guess (0, 0), zero nonlinear functions
        return 2, matrix(starting_point)

    # Zielfunktion
    f = matrix([x[0], f_1(x), f_2(x)]) # Dimension (1,1)

    # Gradient
    Df = matrix([[1, 0], [-2 * x[0], -2 * x[1]], [4 * x[0], 4 * x[1]]]).T # Gradient: Dimension (1,2)

    if z is None:
        return f, Df

    H = z[1] * matrix([[-2.0, 0.0], [0.0, -2.0]]) + z[2] * matrix([[4.0, 0.0], [0.0, 4.0]])
    print(f'f[{np.round(x[0], 3)}, {np.round(x[1], 3)}] = {np.round(f[0], 3)}')
    return f, Df, H

# Inequality Constraints:
#G = matrix([[-1.0, -1.0], [-1.0, 0.0], [0.0, -1.0]]).T
#h = matrix([3.0, 0.0, 0.0])

#print(G.size)
#print(h.size)

## Equality Constraints

#A = matrix([1.0, -1.0]).T
#b = matrix([1.0])

# Solve
solution = solvers.cp(F)

# Optimale Lösung
x_opt = solution['x']
optimal_value = solution['primal objective']

print("Optimal solution (x):", x_opt[0], x_opt[1])
print("Minimal value of the function:", optimal_value)

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 400)  # Range from -3 to 3
y = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x, y)

# Compute f1(x, y) for each (x, y) pair
Z = f1(X, Y)

# Plot the contours for f1(x, y) = 4 and f1(x, y) = 3
plt.figure(figsize=(8, 6))
contours = plt.contour(X, Y, Z, levels=[3, 4], colors=['blue', 'red'], linewidths=2)
plt.clabel(contours, inline=True, fontsize=10)

# Plot the points at (0,0) and (-0.435, 1.5)
plt.scatter(0, 0, color='black', marker='o', label='Point (0,0)', zorder=5)
plt.scatter(x_opt[0], x_opt[1], color='green', marker='x', label=f'Solution ({np.round(x_opt[0], 2)}, {np.round(x_opt[1], 2)})', zorder=5)
plt.scatter(starting_point[0], starting_point[1], color='purple', marker='x', label=f'Start ({starting_point[0]}, {starting_point[1]})', zorder=5)

# Add labels for the points
plt.text(0, 0, '(0, 0)', fontsize=10, ha='right', va='bottom')
#plt.text(x_opt[0], x_opt[1], f'({np.round(x_opt[0], 2)}, {np.round(x_opt[1], 2)})', fontsize=10, ha='left', va='bottom')
#plt.text(starting_point[0], starting_point[1], f'({starting_point[0]}, {starting_point[1]})', fontsize=10, ha='left', va='bottom')

# Plot settings
plt.title(r'Contours of $f_1(x, y) = x^2 + y^2$ with Points')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.axis('equal')  # Keep aspect ratio of the plot equal
plt.legend()
plt.show()