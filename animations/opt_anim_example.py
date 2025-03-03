import numpy as np
from manim import *

## Optimize simple function 2x² + 3x + 1 with gradient descent

# Define the function
def f(x):
    return 2*x**2 + 3*x + 1

# Define the gradient
def df(x):
    return 4*x + 3

# Define the learning rate
lr = 0.02

# Define the initial guess
x = 0

# Define the number of iterations
n_iter = 30

# Perform the optimization
xs = [x]
for i in range(n_iter):
    x = x - lr * df(x)
    xs.append(x)

# Print the result
print(f"Optimal x: {x}")

# Plot the optimization path
import matplotlib.pyplot as plt
plt.plot(xs, [f(x) for x in xs], 'o-')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Optimization path')
plt.show()

# Optimal x: -0.75

class OptimizationManim(Scene):
    def construct(self):
        plot_axes = Axes(
            x_range = [-1, 1, 0.05],
            y_range = [-1, 1, 0.05],
            x_length=9,
            y_length=6,
            axis_config={"numbers_to_include": np.arange(-1, 1+0.2, 0.2), "font_size": 24}
        )