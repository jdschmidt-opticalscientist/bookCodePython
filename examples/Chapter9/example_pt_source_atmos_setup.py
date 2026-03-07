import numpy as np
from scipy.optimize import minimize

# example_pt_source_atmos_setup.py

# 1. Geometry and Basic Parameters
D2 = 0.5           # Obs aperture [m]
wvl = 1e-6         # Wavelength [m]
k = 2 * np.pi / wvl
Dz = 50e3          # Total distance [m]

DROI = 4 * D2
D1 = wvl * Dz / DROI
R = Dz             # Wavefront radius of curvature

# 2. Atmospheric Properties
Cn2 = 1e-16        # Constant structure parameter
r0sw = (0.423 * k**2 * Cn2 * (3/8) * Dz)**(-3/5)
r0pw = (0.423 * k**2 * Cn2 * Dz)**(-3/5)

p = np.linspace(0, Dz, 1000)
dp = p[1] - p[0]
rytov = 0.563 * k**(7/6) * np.sum(Cn2 * (1 - p/Dz)**(5/6) * p**(5/6) * dp)

# 3. Optimization Setup for Screen Strengths
nscr = 11
alpha = np.linspace(0, 1, nscr)
A = np.zeros((2, nscr))
A[0, :] = alpha**(5/3)
A[1, :] = (1 - alpha)**(5/6) * alpha**(5/6)

b = np.array([r0sw**(-5/3), rytov / 1.33 * (k/Dz)**(5/6)])

# Objective function: minimize sum of squared errors
def objective(X):
    return np.sum((A @ X - b)**2)

# Initial guess
x0 = (nscr / 3 * r0sw * np.ones(nscr))**(-5/3)

# Constraints and Bounds
# rmax limits the scintillation per step to keep the simulation valid
rmax = 0.1
bounds = []
for i in range(nscr):
    lower = 0
    if A[1, i] == 0:
        upper = 50**(-5/3) # Large r0 default
    else:
        upper = (rmax / 1.33 * (k/Dz)**(5/6)) / A[1, i]
    bounds.append((lower, upper))

# Perform Optimization
res = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
X = res.x

# 4. Results Check
r0scrn = X**(-3/5)
r0scrn[np.isinf(r0scrn)] = 1e6

bp = A @ X
print(f"Calculated [r0sw, rytov]: {bp[0]**(-3/5):.4f}, {bp[1]*1.33*(Dz/k)**(5/6):.4f}")
print(f"Theoretical [r0sw, rytov]: {r0sw:.4f}, {rytov:.4f}")