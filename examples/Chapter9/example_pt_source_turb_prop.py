import numpy as np
# from wave_prop import ft_sh_phase_screen, ang_spec_multi_prop, circ, corr2_ft

# 1. Setup parameters
l0 = 0.0
L0 = np.inf
nreals = 20  # Number of random realizations

# Locations and distances
zt = np.concatenate(([0], z_planes)) # cumulative distances
Delta_z = np.diff(zt)

# Grid spacing interpolation along the path
alpha = zt / zt[-1]
delta_path = (1 - alpha) * delta1 + alpha * deltan

# 2. Initialization
phz = np.zeros((N, N, nscr))
Uout_avg = np.zeros((N, N), dtype=complex)
MCF2 = np.zeros((N, N), dtype=complex)

# Circular mask for the observation aperture
mask = circ(xn, yn, D2)

# 3. Monte Carlo Loop
for idxreal in range(nreals):
    print(f"Realization {idxreal + 1}/{nreals}")
    
    # Generate random phase screens for each plane
    for idxscr in range(nscr):
        phz_lo, phz_hi = ft_sh_phase_screen(r0scrn[idxscr], N, delta_path[idxscr], L0, l0)
        phz[:, :, idxscr] = phz_lo + phz_hi
        
    # Combine phase screens with the absorbing guard band (sg)
    # sg and pt are inherited from the vacuum script
    phase_factor = sg_stack * np.exp(1j * phz)
    
    # Simulate turbulent propagation
    xn, yn, Uout = ang_spec_multi_prop(pt, wvl, delta1, deltan, z_planes, phase_factor)
    
    # Collimate the beam (remove quadratic curvature)
    Uout *= np.exp(-1j * np.pi / (wvl * R) * (xn**2 + yn**2))
    
    # Accumulate the Mutual Coherence Function (MCF)
    # corr2_ft calculates the spatial correlation of the field
    MCF2 += corr2_ft(Uout, Uout, mask, deltan)

# 4. Modulus of the complex degree of coherence (MCDOC)
mid = N // 2
MCDOC2 = np.abs(MCF2) / np.abs(MCF2[mid, mid])