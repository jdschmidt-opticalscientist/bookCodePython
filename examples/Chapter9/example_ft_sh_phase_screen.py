import numpy as np
import matplotlib.pyplot as plt
from OpticalWavePropSim import ft_sh_phase_screen

# example_ft_sh_phase_screen.m
D = 2.0           # Length of one side of square phase screen [m]
r0 = 0.1          # Fried parameter (coherence diameter) [m]
N = 256           # Grid points per side
L0 = 100.0        # Outer scale [m]
l0 = 0.01         # Inner scale [m]

delta = D / N     # Grid spacing [m]
x_vec = np.arange(-N/2, N/2) * delta

# Generate phase screen components
# phz_lo contains the subharmonic (low frequency) compensation
# phz_hi contains the standard FFT-based high frequencies
phz_lo, phz_hi = ft_sh_phase_screen(r0, N, delta, L0, l0)
phz = phz_lo + phz_hi

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(131)
plt.imshow(phz_hi, extent=[x_vec[0], x_vec[-1], x_vec[0], x_vec[-1]])
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.title("High Freq (FFT only)")

plt.subplot(132)
plt.imshow(phz_lo, extent=[x_vec[0], x_vec[-1], x_vec[0], x_vec[-1]])
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.title("Low Freq (Subharmonics)")

plt.subplot(133)
plt.imshow(phz, extent=[x_vec[0], x_vec[-1], x_vec[0], x_vec[-1]])
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.title("Combined Phase Screen")

plt.tight_layout()
plt.show()