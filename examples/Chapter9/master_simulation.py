import pt_source_atmos_setup
import pt_source_atmos_samp
import propagation_models
import numpy as np
import matplotlib.pyplot as plt

# Setup propagaton scenario
D2 = 0.5    # diameter of the observation aperture [m]
wvl = 1e-6  # optical wavelength [m]
Dz = 50e3   # propagation distance [m]
atm = pt_source_atmos_setup.setup_atmosphere(D2, wvl, Dz)

# analyze sampling constraints
sampParams = pt_source_atmos_samp.analyze_sampling(atm)
N = sampParams['N']
d1 = sampParams['d1']
dn = sampParams['d2']

# Run Vacuum
xn, yn, Uvac = propagation_models.pt_source_vac_prop(atm, N, d1, dn)

# Run Turbulence
xn, yn, mcf_turb = propagation_models.pt_source_turb_prop(atm, N, d1, dn, nreals=50)

# theoretical coherence
mid = N // 2
mcdoc = np.abs(mcf_turb) / np.abs(mcf_turb[mid, mid])
r0 = atm["r0sw"]
MCDOC_th = np.exp(-3.44*(abs(xn[mid,mid:-1])/r0)**(5/3))

#
# Show Coherence
#
plt.figure()

plt.plot(xn[mid,mid:-1]/r0, mcdoc[mid,mid:-1], color='orange', label='Simulation')
plt.plot(xn[mid,mid:-1]/r0, MCDOC_th, 'k-.', label='Theory')
plt.xlim(0, 4)
plt.grid(True, alpha=0.3)
plt.title("Coherence (MCDOC) with Atmosphere")
plt.xlabel("Separation/$r_0$")
plt.ylabel("Spatial Coherence Factor")

plt.show()