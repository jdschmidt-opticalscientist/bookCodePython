import numpy as np
from scipy.optimize import minimize

def setup_atmosphere(D2, wvl, Dz):
    k = 2 * np.pi / wvl
    
    # use sinc to model pt source
    DROI = 4 * D2 # diam of obs-plane region of interest [m]
    D1 = wvl * Dz / DROI # width of central lobe [m]
    R = Dz  # Wavefront radius of curvature [m]

    # Atmospheric Properties
    Cn2 = 1e-16  # structure parameter [m^-2/3], constant
    # SW and PW coherence diameters [m]
    r0sw = (0.423 * k**2 * Cn2 * (3/8) * Dz)**(-3/5)
    r0pw = (0.423 * k**2 * Cn2 * Dz)**(-3/5)
    # log-amplitude variance
    p = np.linspace(0, Dz, 1000)
    dp = p[1] - p[0]
    rytov = 0.563 * k**(7/6) * np.sum(Cn2 * (1 - p/Dz)**(5/6) * p**(5/6) * dp)

    # screen properties
    nscr = 11 # number of screens
    alpha = np.linspace(0, 1, nscr)
    A = np.zeros((2, nscr)) # matrix
    A[0, :] = alpha**(5/3)
    A[1, :] = (1 - alpha)**(5/6) * alpha**(5/6)
    b = np.array([r0sw**(-5/3), rytov / 1.33 * (k/Dz)**(5/6)])

    # initial guess
    x0 = (nscr / 3 * r0sw * np.ones(nscr))**(-5/3)
    # objective function
    def objective(X):
        return np.sum((A @ X - b)**2)
    
    # constraints
    rmax = 0.1 # maximum Rytov number per partial prop
    bounds = []
    for i in range(nscr):
        upper = (rmax / 1.33 * (k/Dz)**(5/6)) / A[1, i] if A[1, i] != 0 else 50**(-5/3)
        bounds.append((0, upper))

    res = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    # check screen r0s
    X = res.x
    #r0scrn = X**(-3/5)
    r0scrn = np.clip(X, 1e-10, None)**(-3/5)
    r0scrn[np.isinf(r0scrn)] = 1e6
    
    print("Individual Phase Screen Strengths:")
    print(f"{'Screen #':<10} | {'r0 [m]':>10}")
    print("-" * 23)
    
    for i, r0_val in enumerate(r0scrn):
        # Using 'g' for the value in case some screens are 
        # effectively infinite (very weak turbulence)
        print(f"Screen {i:<3} | {r0_val:>10.4g}")
    
    print("-" * 23)
    
    # check resulting r0sw & rytov
    bp = A @ X;

    sim_r0 = bp[0]**(-3/5)
    # Calculate Rytov number
    sim_rytov = bp[1] * 1.33 * (Dz/k)**(5/6)
    
    # Calculate Error
    error_r0 = abs(r0sw - sim_r0) / r0sw
    
    # Print a table
    print("\n" + "="*45)
    print(f"{'Metric':<15} | {'Theoretical':>12} | {'Simulated':>12}")
    print("-" * 45)
    print(f"{'r0 (SW)':<15} | {r0sw:>12.4f} | {sim_r0:>12.4f}")
    print(f"{'Rytov Var.':<15} | {rytov:>12.4e} | {sim_rytov:>12.4e}")
    print(f"{'r0 Error':<15} | {'-':>12} | {error_r0:>12.2%}")
    print("="*45 + "\n")

    return {
        "D1": D1, "D2": D2, "wvl": wvl, "Dz": Dz, "R": R,
        "r0sw": r0sw, "rytov": rytov, "r0scrn": r0scrn, "nscr": nscr
    }