import numpy as np
import matplotlib.pyplot as plt
# from wave_prop import rect, myconv, tri

# example_conv_rect_rect.m
N = 64          # number of samples
L = 8.0         # grid size [m]
delta = L / N   # sample spacing [m]
x = np.arange(-N/2, N/2) * delta

w = 2.0         # width of rectangle
A = rect(x / w) # signal A
B = A           # signal B

# Perform discrete convolution using the library function
C = myconv(A, B, delta)

# Continuous (analytic) convolution for comparison
C_cont = w * tri(x / w)

# Visualization
plt.figure()
plt.plot(x, np.real(C), 'bo', label='Discrete Convolution')
plt.plot(x, C_cont, 'r-', label='Analytic (Triangle)')
plt.legend()
plt.title('Convolution of two Rect functions')
plt.grid(True)
plt.show()