import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2*np.pi, 2*np.pi, 100)

# Plot sin, cos, tan, cot, arccot
def arccot(x):
    if x < 0:
        return np.arctan(1/x) + np.pi
    return np.arctan(1/x)

arccoty = np.array([arccot(xi) for xi in x])

#plt.plot(x, np.sin(x), label='sin')
#plt.plot(x, np.cos(x), label='cos')
#plt.plot(x, 1/np.tan(x), label='cot')
#plt.plot(x, arccoty, label='arccot')

plt.plot(x, np.cos(arccoty), label=r'cos(arccot)')
plt.plot(x, x/np.sqrt(x**2 + 1), label=r'x/sqrt(x^2 + 1)')
plt.plot(x, 1/np.sqrt(1/x**2 + 1), label=r'1/sqrt(1/x^2 + 1)')

plt.legend()

# Plot vertical line and horizontal line in (0,0)
plt.axvline(0, color='black', lw=0.5)
plt.axhline(0, color='black', lw=0.5)

plt.ylim(-6, 6)
plt.xlim(-6, 6)

plt.show()