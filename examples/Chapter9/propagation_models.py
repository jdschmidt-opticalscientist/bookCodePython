import numpy as np
import matplotlib.pyplot as plt
from OpticalWavePropSim import ang_spec_multi_prop, ft_sh_phase_screen, circ, corr2_ft

def _create_source(params, N, delta1):
    """Internal helper to build the band-limited point source."""
    wvl  = params['wvl']
    R = params['R']
    D1 = params['D1']
    k = 2 * np.pi / wvl
    # coordinates
    x1vec = np.arange(-N/2, N/2) * delta1
    x1, y1 = np.meshgrid(x1vec, x1vec)
    r1 = np.sqrt(x1**2 + y1**2)
    
    # point source
    pt = (np.exp(-1j * k / (2 * R) * r1**2) / D1**2 * np.sinc(x1/D1) * np.sinc(y1/D1) * np.exp(-(r1 / (4 * D1))**2))
    
    # 16th order Super-Gaussian absorbing mask
    sg = np.exp(-(x1 / (0.47 * N * delta1))**16) * np.exp(-(y1 / (0.47 * N * delta1))**16)
    
    return pt, sg, x1, y1

def pt_source_vac_prop(params, N=512, delta1=0.01, deltan=0.01):
    """Pure vacuum propagation for baseline reference."""
    wvl  = params['wvl']
    R = params['R']
    Dz = params['Dz']
    n = params['nscr'] # number of planes
    # point source
    pt, sg, _, _ = _create_source(params, N, delta1)
    
    # partial prop planes
    z = np.arange(1, n) * Dz / (n - 1)
    
    # simulate vacuum propagation
    t = np.repeat(sg[:, :, np.newaxis], n, axis=2)
    xn, yn, Uvac = ang_spec_multi_prop(pt, wvl, delta1, deltan, z, t)
    # collimate the beam
    Uvac *= np.exp(-1j * np.pi / (wvl * R) * (xn**2 + yn**2))
    
    #
    # show the vacuum-propagated irradiance and phase
    #
    plt.figure(figsize=(10, 4))

    plt.subplot(121)
    im1 = plt.imshow(np.abs(Uvac)**2, extent=[xn[1,1], xn[1,-1], yn[1,1], yn[-1,1]])
    plt.title("Vacuum Irradiance")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.colorbar(im1, label='Irradiance')

    plt.subplot(122)
    im1 = plt.imshow(np.angle(Uvac), extent=[xn[1,1], xn[1,-1], yn[1,1], yn[-1,1]])
    plt.title("Vacuum Phase")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.colorbar(im1, label='Phase [rad]')
    plt.tight_layout()
    plt.show()
    
    return xn, yn, Uvac

def pt_source_turb_prop(params, N=512, delta1=0.01, deltan=0.01, nreals=1):
    """Full Monte Carlo turbulent propagation."""
    wvl = params['wvl']
    R = params['R']
    Dz = params['Dz']
    n = params['nscr']
    r0scrn = params['r0scrn']
    D2 = params['D2']
    
    # point source
    pt, sg, _, _ = _create_source(params, N, delta1)
    
    l0 = 1e-20     # inner scale [m]
    L0 = 1e20   # outer scale [m]
    
    z = np.arange(1, n) * Dz / (n - 1)
    zt = np.concatenate(([0], z)) # propagation plane locations
    # grid spacings
    alpha = zt/zt[-1]
    delta = (1 - alpha) * delta1 + alpha * deltan
    
    # observation-plane coordinates
    xnvec = np.arange(-N/2, N/2) * deltan
    xn, yn = np.meshgrid(xnvec, xnvec)
    
    # initialize array for phase screens
    phz = np.zeros((N, N, n), dtype=complex)
    # initialize arrays for propagated fields,
    # aperture mask, and MCF
    Uout = np.zeros((N, N), dtype=complex)
    mask = circ(xn/D2, yn/D2, 1)
    MCF2 = np.zeros((N, N), dtype=complex)
    for idxreal in range(nreals): # loop over realizations
        
        for idxscr in range(n): # loop over screens
            lo, hi = ft_sh_phase_screen(r0scrn[idxscr], N, delta[idxscr], L0, l0)
            phz[:,:,idxscr] = sg * np.exp(1j * (lo + hi))
        # simulate turbulent propagation    
        xn, yn, Uout = ang_spec_multi_prop(pt, wvl, delta1, deltan, z, phz)
        # collimate the beam
        Uout *= np.exp(-1j * np.pi / (wvl * R) * (xn**2 + yn**2))
        # accumulate realizations of the MCF
        MCF0, maskCorr = corr2_ft(Uout, Uout, mask, deltan)
        MCF2 += MCF0
        
    # modulus of the complex degree of coherence
    mid = N // 2
    MCF2 = np.abs(MCF2) / np.abs(MCF2[mid, mid])
    
    #
    # show last realization
    #
    plt.figure(figsize=(10, 4))

    plt.subplot(121)
    im1 = plt.imshow(np.abs(Uout)**2, extent=[xn[1,1], xn[1,-1], yn[1,1], yn[-1,1]])
    plt.title("Turbulent Irradiance")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.colorbar(im1, label='Irradiance')

    plt.subplot(122)
    im1 = plt.imshow(np.angle(Uout), extent=[xn[1,1], xn[1,-1], yn[1,1], yn[-1,1]])
    plt.title("Turbulent Phase")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.colorbar(im1, label='Phase')
    plt.tight_layout()
    plt.show()

    return xn, yn, MCF2