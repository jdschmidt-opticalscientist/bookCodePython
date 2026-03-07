import numpy as np
import matplotlib.pyplot as plt
from OpticalWavePropSim import rect, myconv, tri

# example_conv_rect_rect.m
N = 64          # number of samples
L = 8.0         # grid size [m]
delta = L / N   # sample spacing [m]
x = np.arange(-N/2, N/2) * delta

w = 2.0         # width of rectangle
A = rect(x / w) # signal A
B = A           # signal B

# Perform discrete convolution
C = myconv(A, B, delta)

# Continuous convolution
C_cont = w * tri(x / w)

# Visualization
plt.figure(figsize=(10, 8))

# Plot (a): Signal A
plt.subplot(2, 2, 1)
plt.plot(x, A, 'bx')
plt.title('(a) $A(x)$')
plt.grid(True)
plt.ylabel('Amplitude')

# Plot (b): Signal B
plt.subplot(2, 2, 2)
plt.plot(x, B, 'gx')
plt.title('(b) $B(x)$')
plt.grid(True)

# Plot (c): Convolution Result (Spanning the bottom row)
plt.subplot(2, 1, 2)
plt.plot(x, C_cont, 'rs-', label='Analytic')
plt.plot(x, np.real(C), 'bx', label='Numerical')
plt.title('(c) $A(x) \otimes B(x)$')
plt.xlabel('x [m]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout() # Adjusts spacing so titles don't overlap
plt.show()