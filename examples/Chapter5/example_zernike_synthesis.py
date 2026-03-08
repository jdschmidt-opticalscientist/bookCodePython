import numpy as np
import matplotlib.pyplot as plt
from OpticalWavePropSim import circ, ft_phase_screen, zernike

N = 40        # number of grid points per side
L = 2         # total size of the grid [m]
delta = L / N # grid spacing [m]
# Cartesian & polar coordinates
x_vec = np.arange(-N/2, N/2) * delta
x, y = np.meshgrid(x_vec, x_vec)
r = np.sqrt(x**2 + y**2)
theta = np.arctan2(y, x)
# unit circle aperture
ap = circ(x, y, 2)
# indices of grid points in aperture
idxAp = ap.astype(bool) # Equivalent to logical(ap)
# create atmospheric phase screen
r0 = L / 20
screen = ft_phase_screen(r0, N, delta, np.inf,1e-20) / (2 * np.pi) * ap
W = screen[idxAp] # perform linear indexing

# analyze screen
nModes = 100 # number of Zernike modes
# create matrix of Zernike polynomial values
Z = np.zeros((W.size, nModes))
for idx in range(1, nModes + 1):
    temp = zernike(idx, r, theta) # Ensure this handles Noll 1-based indexing
    Z[:, idx-1] = temp[idxAp]
# compute mode coefficients
A, _, _, _ = np.linalg.lstsq(Z, W, rcond=None)
# synthesize mode-limited screen
W_prime = Z @ A
# reshape mode-limited screen into 2-D for display
scr = np.zeros((N, N))
scr[idxAp] = W_prime

# Visualization
c_min = np.min(W)
c_max = np.max(W)

fig = plt.figure(figsize=(10, 16))

# Original Screen (Top Center)
ax_orig = plt.subplot2grid((4, 2), (0, 0), colspan=2)
im_orig = ax_orig.imshow(screen, extent=[-L/2, L/2, -L/2, L/2], 
                         cmap='jet', vmin=c_min, vmax=c_max)
ax_orig.set_title("Original Screen")
plt.colorbar(im_orig, ax=ax_orig, label='Waves [$\lambda$]')

# Reconstructions using cumulative modes
mode_list = [3, 16, 36, 100]
plot_locs = [(1, 0), (1, 1), (2, 0), (2, 1)]

for i, n in enumerate(mode_list):
    # Sum modes 1 through n
    W_prime = Z[:, :n] @ A[:n]
    
    scr = np.zeros((N, N))
    scr[idxAp] = W_prime
    
    ax = plt.subplot2grid((4, 2), plot_locs[i])
    # Apply the same vmin and vmax here
    im = ax.imshow(scr, extent=[-L/2, L/2, -L/2, L/2], 
                   cmap='jet', vmin=c_min, vmax=c_max)
    ax.set_title(f"Modes 1 to {n}")
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()