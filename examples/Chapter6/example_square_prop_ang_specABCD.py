import numpy as np
# from wave_prop import rect, ang_spec_propABCD

# example_square_prop_ang_specABCD.m
N = 1024
L = 1e-2
delta1 = L / N
D = 2e-3
wvl = 1e-6
k = 2 * np.pi / wvl
Dz = 1.0
f = np.inf  # Collimated source

# Source plane coordinates
vec1 = np.arange(-N/2, N/2) * delta1
x1, y1 = np.meshgrid(vec1, vec1)

# Square aperture
ap = rect(x1 / D) * rect(y1 / D)

# Observation scaling
delta2 = (wvl * Dz) / (N * delta1)

# Construct ABCD Matrix
# Propagation (D) * Lens/Curvature (f)
M_prop = np.array([[1, Dz], [0, 1]])
M_lens = np.array([[1, 0], [-1/f, 1]]) if f != np.inf else np.eye(2)
ABCD = M_prop @ M_lens

# Numerical propagation via ABCD method
x2, y2, Uout = ang_spec_propABCD(ap, wvl, delta1, delta2, ABCD)