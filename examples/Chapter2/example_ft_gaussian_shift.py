# example_ft_gaussian_shift.py -----------------------------------

import numpy as np
import matplotlib.pyplot as plt
from OpticalWavePropSim import ft 

L = 10.0      # spatial extent of the grid
N = 64        # number of samples
delta = L / N # sample spacing
x = np.arange(-N/2, N/2) * delta
x0 = 5 * delta
f = np.arange(-N/2, N/2) / (N * delta)
a = 1.0
# sampled function & its DFT
g_samp = np.exp(-np.pi * a * (x - x0)**2)
g_dft = ft(g_samp, delta)
# analytic function and its continuous FT
M = 1024
x_cont = np.linspace(x[0], x[-1], M)
f_cont = np.linspace(f[0], f[-1], M)
g_cont = np.exp(-np.pi*a*(x_cont-x0)**2);
g_ft_cont = np.exp(-1j * 2 * np.pi * x0 * f_cont) * np.exp(-np.pi * f_cont**2 / a) / a

# --- Plotting ---
plt.figure(figsize=(10, 6))

# Plot the continuous analytic FT
plt.plot(f_cont, np.real(g_ft_cont), 'r-', label='Analytic FT (Real)', linewidth=1.5)
plt.plot(f_cont, np.imag(g_ft_cont), 'g-', label='Analytic FT (Imag)', linewidth=1.5)
plt.plot(f_cont, np.abs(g_ft_cont), 'b-', label='Anayltic FT (Abs)', linewidth=1.5)

# Plot the DFT result as discrete points
plt.plot(f, np.real(g_dft), 'rx', label='DFT (Real)', markersize=4)
plt.plot(f, np.imag(g_dft), 'gx', label='DFT (Imag)', markersize=4)
plt.plot(f, np.abs(g_dft), 'bx', label='DFT (Abs)', markersize=4)

plt.title('Fourier Transform of a Gaussian: Numerical vs. Analytic')
plt.xlabel('Frequency $f$ [cycles/m]')
plt.ylabel('$G(f)$')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()