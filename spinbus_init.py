import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

#units
hbar = 0.6582119569       # micro-eV * ns
mu_B = 57.88381806        # micro-eV / Tesla
g = 2.0                   # approximately for Si

def mT_to_ueV(B_mT):
    """Convert magnetic field in mT to energy in micro-eV."""
    return g * mu_B * (B_mT / 1000.0)  # Convert mT to Tesla

B_parallel=mT_to_ueV(20.0)
dB_parallel=mT_to_ueV(1.3)
dB_perp=mT_to_ueV(0.3)

# Exchange J(epsilon)

h = 2.0 * np.pi * hbar       # micro-eV * ns

# Dial-fit reconstruction:
# 160 MHz = 0.160 ns^-1
J0 = h * 0.160               # micro-eV
eps0 = 0.27241 #mV
eps_start = 0.8 #mV
eps_end = -4.0 #mV

T_pulse = 210.0              # ns
T_idle = 10.0                # ns
T_ramp = T_pulse - T_idle    # 200 ns


def epsilon_of_t(t):
    if t <= T_ramp:
        return eps_start + (eps_end - eps_start) * (t / T_ramp)

    return eps_end


def J_of_epsilon(epsilon):
    return J0 * np.exp(epsilon/eps0)

def crossing_equation(epsilon):
    return J_of_epsilon(epsilon) - B_parallel

result = root_scalar(crossing_equation, bracket=[eps_end, eps_start], method='brentq')
eps_cross = result.root
print("Crossing epsilon =", eps_cross, "mV")
eps_jump_amp = 0.18    # mV

eps_jump_high = eps_cross + eps_jump_amp / 2.0
eps_jump_low  = eps_cross - eps_jump_amp / 2.0
continuous_drop = (eps_start - eps_end) - eps_jump_amp

slow_rate = continuous_drop / T_ramp    # mV/ns
# Time when we reach the upper edge of the jump
t_jump = (eps_start - eps_jump_high) / slow_rate


def epsilon_of_t(t):

    if t < t_jump:

        # Slow ramp before crossing
        return eps_start - slow_rate * t

    elif t <= T_ramp:

        # Instantaneous jump has occurred.
        # Continue slow ramp from lower side.
        return eps_jump_low - slow_rate * (t - t_jump)

    else:

        # Final idle period
        return eps_end

# J(t) and dJ/dt diagnostics

times = np.linspace(0, T_pulse, 2000)

epsilon_t = np.array([
    epsilon_of_t(t) for t in times
])

J_t = np.array([
    J_of_epsilon(eps) for eps in epsilon_t
])

plt.figure()

plt.plot(times, J_t, label='J(t)')
plt.axhline(
    B_parallel,
    linestyle='--',
    label='B_parallel'
)

plt.xlabel('Time (ns)')
plt.ylabel('Energy (micro-eV)')
plt.title('Exchange Interaction J(t)')
plt.legend()
plt.tight_layout()
plt.show()

dJ_dt = np.gradient(
    J_t,
    times,
    edge_order=2
)

plt.figure()

plt.plot(times, dJ_dt)

plt.xlabel('Time (ns)')
plt.ylabel('dJ/dt (micro-eV/ns)')
plt.title('Rate of Change of Exchange Interaction')
plt.tight_layout()
plt.show()

cross_index = np.argmin(
    np.abs(J_t - B_parallel)
)

t_cross = times[cross_index]
J_cross = J_t[cross_index]
sweep_rate_cross=abs(dJ_dt[cross_index])
#S-T- coupling
V_STm = dB_perp/(2.0*np.sqrt(2.0))
#Landau Zener Parameter
LZ_exponent = (2.0 * np.pi * V_STm**2) / (hbar * sweep_rate_cross)

P_diabatic = np.exp(-LZ_exponent)
P_adiabatic = 1.0 - P_diabatic

print("S-T- crossing time =", t_cross, "ns")
print("J at crossing =", J_cross, "micro-eV")
print("|dJ/dt| at crossing =", sweep_rate_cross, "micro-eV/ns")
print("S-T- coupling V =", V_STm, "micro-eV")
print("Landau-Zener exponent =", LZ_exponent)
print("Estimated diabatic probability =", P_diabatic)
print("Estimated adiabatic-transition probability =", P_adiabatic)

# Hamiltonian basis = {|T0>, |S>, |T->}

print("J(start) =", J_of_epsilon(eps_start))
print("J(end)   =", J_of_epsilon(eps_end))
print("B energy =", B_parallel)

def H(t):
    epsilon = epsilon_of_t(t)
    J = J_of_epsilon(epsilon)
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

energies = np.array([
    np.linalg.eigvalsh(H(t)) for t in times
])
plt.figure()
plt.plot(times, energies[:,0], label='E1')
plt.plot(times, energies[:,1], label='E2')
plt.plot(times, energies[:,2], label='E3')
plt.xlabel('Time (ns)')
plt.ylabel('Energy (micro-eV)')
plt.title('Instantaneous Energy Levels')
plt.legend()
plt.tight_layout()
plt.show()