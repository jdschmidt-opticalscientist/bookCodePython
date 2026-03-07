import numpy as np
import matplotlib.pyplot as plt
# from wave_prop import circ, ft_phase_screen, zernike

# example_zernike_synthesis.m
N = 40
L = 2.0
delta = L / N

# Coordinates
vec = np.arange(-N/2, N/2) * delta
x, y = np.meshgrid(vec, vec)
r = np.sqrt(x**2 + y**2)
theta = np.arctan2(y, x)

# Aperture and indexing
ap = circ(x, y, 2.0)
idx_ap = ap > 0

# Create phase screen (converted from phase to waves by dividing by 2pi)
r0 = L / 20
screen_full = ft_phase_screen(r0, N, delta, np.inf, 0) / (2 * np.pi) * ap
W = screen_full[idx_ap]

# Analyze Screen
n_modes = 100
# Initialize Zernike matrix: Rows = pixels in aperture, Cols = Zernike modes
Z = np.zeros((np.sum(idx_ap), n_modes))

# You'll need a way to map idx to (n, m). Here's a placeholder for the logic:
# for idx in range(1, n_modes + 1):
#     temp = zernike(idx, r, theta, zern_index_map)
#     Z[:, idx-1] = temp[idx_ap]

# Compute coefficients via Least Squares
A, _, _, _ = np.linalg.lstsq(Z, W, rcond=None)

# Synthesize mode-limited screen
W_prime = Z @ A

# Reshape back to 2D
scr = np.zeros((N, N))
scr[idx_ap] = W_prime

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(121); plt.imshow(screen_full); plt.title("Original Turbulence")
plt.subplot(122); plt.imshow(scr); plt.title(f"Reconstructed ({n_modes} modes)")
plt.show()