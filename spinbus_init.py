import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#units
hbar = 0.6582119569       # micro-eV * ns
mu_B = 57.88381806        # micro-eV / Tesla
g = 2.0                   # approximately for Si

def mT_to_ueV(B_mT):
    """Convert magnetic field in mT to energy in micro-eV."""
    return g * mu_B * (B_mT / 1000.0)  # Convert mT to Tesla

B_parallel=mT_to_ueV(20.0)
dB_parallel=mT_to_ueV(1.0)
dB_perp=mT_to_ueV(0.3)

T_pulse=200.0 #ns

J_start = 10.0            # micro-eV
J_end = 0.0               # micro-eV

def J_of_t(t):
    return J_start + (J_end - J_start) * (t / T_pulse)

# Hamiltonian basis = {|T0>, |S>, |T->}

def H(t):
    J = J_of_t(t)
    return np.array([[0.0, dB_parallel/2, 0.0],
                     [dB_parallel/2.0, -J, dB_perp/(2.0*np.sqrt(2.0))],
                     [0.0, dB_perp/(2.0*np.sqrt(2.0)), -B_parallel]]
                     ,dtype=complex)

psi0 = np.array([
    0.0,
    1.0,
    0.0
], dtype=complex)


# Target state |down up>
# = (|T0> - |S>) / sqrt(2)

psi_target = np.array([
    1.0,
   -1.0,
    0.0
], dtype=complex) / np.sqrt(2.0)

# Time-dependent Schrodinger equation

def schrodinger(t, psi):
    return (-1j / hbar) * H(t) @ psi

times=np.linspace(0,T_pulse,2000)
solution=solve_ivp(schrodinger, [0, T_pulse], psi0, t_eval=times, method='RK45', rtol=1e-8, atol=1e-10)
print("Solver success:", solution.success)
print("Solver message:", solution.message)
psi_t = solution.y

#Populations
P_T0=np.abs(psi_t[0])**2
P_S=np.abs(psi_t[1])**2
P_Tm=np.abs(psi_t[2])**2

#Fidelity with target state
psi_final = psi_t[:, -1]
F=np.abs(np.vdot(psi_target, psi_final))**2
norm=np.vdot(psi_final, psi_final).real

print("Final normalization=", norm)
print("Initialization fidelity=", F)

plt.plot(times, P_T0, label='P(T0)')
plt.plot(times, P_S, label='P(S)')
plt.plot(times, P_Tm, label='P(T-)')
plt.xlabel('Time (ns)') 
plt.ylabel('Population')
plt.title('Spin Bus Initialization Dynamics') 
plt.legend()
plt.tight_layout()
plt.show()