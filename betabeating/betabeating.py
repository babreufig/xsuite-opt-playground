import numpy as np
from scipy.optimize import minimize

bb = 0.8   # Example value for bb
ba = 0.7   # Example value for ba
phib = np.pi * 1.2  # Example value for phib (45 degrees in radians)
phia = np.pi * 1.4  # Example value for phia (30 degrees in radians)
aa = 1.1
ab = 1.3  # Example value for ab
k1l = 0.9

M_AB = np.array([[np.sqrt(bb/ba) * (np.cos(phib - phia) + aa * np.sin(phib - phia)),
                  np.sqrt(bb * ba) * np.sin(phib - phia)],
                 [((aa - ab) * np.cos(phib - phia) - (1 + aa * ab) * np.sin(phib - phia)) / np.sqrt(bb * ba),
                  (np.sqrt(ba/bb) * (np.cos(phib - phia) - ab * np.sin(phib - phia)))]])

M_k1l = np.array([[1, 0], [-k1l, 1]])

M_calc = M_AB @ M_k1l

def objective(vars):
    abp, bbp, phibp = vars

    M_ABP = np.array([[np.sqrt(bbp/ba) * (np.cos(phibp - phia) + aa * np.sin(phibp - phia)),
                  np.sqrt(bbp * ba) * np.sin(phibp - phia)],
                 [((aa - abp) * np.cos(phibp - phia) - (1 + aa * abp) * np.sin(phibp - phia)) / np.sqrt(bbp * ba),
                  (np.sqrt(ba/bbp) * (np.cos(phibp - phia) - abp * np.sin(phibp - phia)))]])

    diff = M_ABP - M_calc

    return np.linalg.norm(diff, ord='fro')

arccotdisc = np.cos(phib - phia) / np.sin(phib - phia) - k1l * ba
adder = 0
#if arccotdisc < 0:
    #adder = np.pi # to use arctan instead of arccot with arccot x = arctan(1/x) + adder

phibp = np.arctan(1 / (np.cos(phib - phia) / np.sin(phib - phia) - (k1l * ba))) + adder + phia
bbp = bb * np.sin(phib - phia)**2 / np.sin(phibp - phia)**2
abp = (np.sin(phibp - phia) * np.cos(phibp - phia) - np.cos(phib - phia) * np.sin(phib - phia) + ab * np.sin(phib - phia)**2)\
        / np.sin(phibp - phia)**2

initial_guess = [abp, bbp, phibp]

M_ABP = np.array([[np.sqrt(bbp/ba) * (np.cos(phibp - phia) + aa * np.sin(phibp - phia)),
                  np.sqrt(bbp * ba) * np.sin(phibp - phia)],
                 [((aa - abp) * np.cos(phibp - phia) - (1 + aa * abp) * np.sin(phibp - phia)) / np.sqrt(bbp * ba),
                  (np.sqrt(ba/bbp) * (np.cos(phibp - phia) - abp * np.sin(phibp - phia)))]])

print(f"Initial guess: {initial_guess}")
print(f"Initial difference: {M_ABP - M_calc}")

print(abp, bbp, phibp)

# Solve with SciPy minimize
res = minimize(objective, initial_guess, tol=1e-12)
print(res.x)

abp_sol, bbp_sol, phibp_sol = res.x

print(objective([abp, bbp, phibp]))
print(objective([abp_sol, bbp_sol, phibp_sol - 2*np.pi]))

print(abp_sol - abp, bbp_sol - bbp, phibp_sol - phibp)