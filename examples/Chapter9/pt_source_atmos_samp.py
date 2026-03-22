import numpy as np
import matplotlib.pyplot as plt

def analyze_sampling(params, c=2):
    D1 = params['D1']
    D2 = params['D2']
    wvl = params['wvl']
    Dz = params['Dz']
    r0sw = params['r0sw']
    R = params['R']
    
    D1p = D1 + c * wvl * Dz / r0sw
    D2p = D2 + c * wvl * Dz / r0sw

    delta1 = np.linspace(1e-6, 1.1 * wvl * Dz / D2p, 100)
    deltan = np.linspace(1e-6, 1.1 * wvl * Dz / D1p, 100)
    
    # constraint 1
    deltan_max = -D2p / D1p * delta1 + wvl * Dz / D1p
    # constraint 3
    d2min3 = (1 + Dz / R) * delta1 - wvl * Dz / D1p
    d2max3 = (1 + Dz / R) * delta1 + wvl * Dz / D1p
    delta1_grid, deltan_grid = np.meshgrid(delta1, deltan)
    # constraint 2
    N2 = (wvl * Dz + D1p * deltan_grid + D2p * delta1_grid) / (2 * delta1_grid * deltan_grid)
    
    # constraint 4
    d1 = 10e-3
    d2 = 10e-3
    N = 512
    zmax = min(d1, d2)**2 * N / wvl
    nmin = np.ceil(Dz / zmax) + 1

    #
    # Create Figure
    #
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the filled contour
    levels = np.linspace(1, 14, 14)
    cf = ax.contourf(delta1_grid * 1e3, deltan_grid * 1e3, np.log2(N2), levels=levels, cmap='jet')
    cs = ax.contour(delta1_grid * 1e3, deltan_grid * 1e3, np.log2(N2), levels=levels, colors='black', linewidths=0.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
    
    # Add the colorbar
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label('$\log_2(N)$')
    
    # Overlay the 1D lines
    ax.plot(delta1 * 1e3, deltan_max * 1e3, 'k--', label='Constraint 1', linewidth=2, zorder=2)
    ax.plot(delta1 * 1e3, d2min3 * 1e3, 'g-.', label='Constraint 3 Min', linewidth=2, zorder=2)
    ax.plot(delta1 * 1e3, d2max3 * 1e3, 'b-.', label='Constraint 3 Max', linewidth=2, zorder=2)

    # Set plot details using the ax object
    ax.set_xlabel("$\delta_1$ [mm]")
    ax.set_ylabel("$\delta_n$ [mm]")
    ax.set_title("Constraints 1, 2, & 3")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits to focus on the interesting region
    ax.set_xlim([delta1.min()*1e3, delta1.max()*1e3])
    ax.set_ylim([deltan.min()*1e3, deltan.max()*1e3])
    
    ax.scatter(d1*1e3, d2*1e3, 
               marker='x', s=100, color='white', 
               facecolor='black', linewidths=3, 
               zorder=5, label='Chosen $(\delta_1, \delta_n)$')

    # Update legend to include the point
    ax.legend(loc='upper right', frameon=True, framealpha=0.8)

    plt.show()

    return {"d1": d1, "d2": d2, "N": N, "zmax": zmax, "nmin": nmin}