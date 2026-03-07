import numpy as np
import matplotlib.pyplot as plt
# from wave_prop import rect, derivative_ft

# example_derivative_ft.m
N = 64          # number of samples
L = 6.0         # grid size [m]
delta = L / N   # grid spacing [m]
x = np.arange(-N/2, N/2) * delta

w = 3.0         # size of window [m]
window = rect(x / w) 
g = (x**5) * window # function to differentiate

# Discrete derivatives using FT property
# We take the real part as the input is real
gp_samp = np.real(derivative_ft(g, delta, 1)) * window
gpp_samp = np.real(derivative_ft(g, delta, 2)) * window

# Analytic derivatives (within the window)
gp = 5 * (x**4) * window
gpp = 20 * (x**3) * window

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(x, gp_samp, 'bo', label='FT Derivative')
plt.plot(x, gp, 'r-', label='Analytic')
plt.title("First Derivative ($5x^4$)")
plt.legend()

plt.subplot(122)
plt.plot(x, gpp_samp, 'go', label='FT Derivative')
plt.plot(x, gpp, 'r-', label='Analytic')
plt.title("Second Derivative ($20x^3$)")
plt.legend()
plt.show()