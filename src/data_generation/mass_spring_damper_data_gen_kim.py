# %% default packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import signal
from sklearn.model_selection import train_test_split
from scipy.stats import qmc

# set up matplotlib
plt.rcParams.update({
    "text.usetex": False,
})


# %% define functions
def create_mass_spring_damper_system(n_mass, mass_vals=1, damp_vals=1, stiff_vals=1, input_vals=None, system='ph'):
    """
    Creates a mass-spring-damper system
    :param n_mass: number of masses, i.e. second order system size (integer value)
    :param mass_vals: mass values either (n_mass,) array or scalar value (same value applied to all masses)
    :param damp_vals: damping values either (n_mass,) array or scalar value (same value applied to all dampers)
    :param stiff_vals: stiffness values either (n_mass,) array or scalar value (same value applied to all springs)
    :param input_vals: array of size (number of inputs,) creates an array of size (n_mass,len(input_vals)) with ones at indices from input_vals (index of excited mass)
    :param system: system output matrices, string with either {'ph'}, '2nd', or 'ss' for port-Hamiltonian, second order or state-space matrices
    :return:

    """

    # create mass matrix
    if isinstance(mass_vals, (list, tuple, np.ndarray)):
        M = np.diag(mass_vals)
    else:  # scalar value
        M = np.eye(n_mass) * mass_vals

    # create damping matrix
    if isinstance(mass_vals, (list, tuple, np.ndarray)):
        D = np.diag(damp_vals)
    else:  # scalar value
        D = np.eye(n_mass) * damp_vals

    # create stiffness matrix
    if not isinstance(stiff_vals, (list, tuple, np.ndarray)):
        # scalar value
        stiff_vals = (np.ones(n_mass) * stiff_vals)
    K = np.zeros((n_mass, n_mass))
    K[:, :] = np.diag(stiff_vals[:])
    K[1:, 1:] += np.diag(stiff_vals[:-1])
    K += -np.diag(stiff_vals[:-1], -1)
    K += -np.diag(stiff_vals[:-1], 1)

    # create input vector
    if input_vals is not None:
        if isinstance(input_vals, int):
            # single input
            assert input_vals <= n_mass
            B_2nd = np.zeros((n_mass, 1))
            B_2nd[input_vals, 0] = 1
        else:
            assert max(list(input_vals)) <= n_mass
            B_2nd = np.zeros((n_mass, len(input_vals)))
            B_2nd[input_vals, np.arange(len(input_vals))] = 1
    else:
        B_2nd = np.zeros((n_mass))

    # create state-space system
    M_inv = np.linalg.inv(M)
    A = np.block([[np.zeros((n_mass, n_mass)), np.eye(n_mass)], [-M_inv @ K, -M_inv @ D]])

    # scale input with M
    B = np.concatenate((np.zeros(B_2nd.shape), M_inv @ B_2nd), axis=0)

    # convert to port-Hamiltonian system
    J = np.diag(np.ones(n_mass), n_mass)
    J += -np.diag(np.ones(n_mass), -n_mass)

    R = linalg.block_diag(np.zeros((n_mass, n_mass)), D)

    Q = linalg.block_diag(K, np.linalg.inv(M))

    # B_ph cancels out M with M_inv due to momentum description
    B_ph = np.concatenate((np.zeros(B_2nd.shape), B_2nd), axis=0)

    if system == 'ph':
        return J, R, Q, B_ph
    elif system == '2nd':
        return M, D, K, B_2nd
    elif system == 'ss':
        return A, B, M
    else:
        raise Exception("system input not known. Choose either 'ph','2nd' or 'ss' ")

def mass_spring_damper_ode(x,t,A,B,u):
    if callable(u):
        # u is a lambda function u(t)
        if np.isscalar(u(t)) or np.ndim(u(t))==0:
            dxdt = A@x + np.squeeze(B*u(t))
        else:
            dxdt = A@x + np.squeeze(B@u(t))
    else:
        # u does not depend on t
        if np.isscalar(u):
            dxdt = A@x + B*u
        else:
            dxdt = A@x + B@u
    return dxdt

# %% Script parameters
n_trajectories = 100
train_test_split_ratio = 0.5

# %% Model parameters
# values from [MorandinNicodemusUnger22]
n_mass = 3 # 100
mass_vals = 4
damp_vals = 1
stiff_vals = 4
input_vals = 0 # None (autonomous) | 0 (first mass excitation - SISO)
use_single_input_traj = False
# input functions
def u_mult(t, amp, omega):
    return np.exp(-t/2)*amp*np.sin(omega*t**2) # in publication [MorandinNicodemusUnger22]
t = np.linspace(0, 10, 500) # time vector

# %% input functions
lhs = qmc.LatinHypercube(d=2, seed=123)
lhs_values = lhs.random(n_trajectories)
# parameter boundaries
amp_min = 0.5
amp_max = 2
omega_min = 0.5
omega_max = 2
# scale lhs values to parameter boundaries
amp = amp_min + lhs_values[:,0]*(amp_max-amp_min)
omega = omega_min + lhs_values[:,1]*(omega_max-omega_min)
# create input functions
u = [u_mult(t, amp_, omega_) for amp_, omega_ in zip(amp,omega)]  # noqa: B905

plt.figure()
for u_ in u:
    plt.plot(t,u_)
plt.xlabel('t (s)')
plt.title('input u(t)')
plt.legend(loc='best')
plt.show(block=False)

# %% LTI system
system_descr = 'ss' # state space representation
A, B, M = create_mass_spring_damper_system(n_mass, mass_vals=mass_vals, damp_vals=damp_vals, stiff_vals=stiff_vals,
                                           input_vals=input_vals, system=system_descr)
# use all states as output
C = np.ones([2*n_mass, 1]).T
D = np.zeros((B.shape[1], C.shape[0])) # no feedthrough
sys = signal.lti(A, B, C, D)

# %% Solve system
# initial conditions
x_init_state = np.zeros((2*n_mass))   # zero full state initial condition
x = np.array([signal.lsim(sys, u_, t, X0=x_init_state, interp=True)[2] for u_ in u])

# %% Create training and test data
# create training and test data
x_train, x_test, u_train, u_test = train_test_split(x, u, train_size=train_test_split_ratio, random_state=123)

# plot training and test data of first mass
plt.figure()
for x_ in x_train:
    plt.plot(t, x_[:,0], color='C0')
for x_ in x_test:
    plt.plot(t, x_[:,0], color='C1')
plt.xlabel('t (s)')
plt.title('position x(t)')
plt.show(block=False)

# %% export generated data
# np.save("../../data/SISO_three-masses/x_train",x_train)
# np.save("../../data/SISO_three-masses/x_test",x_test)
# np.save("../../data/SISO_three-masses/u_train",u_train)
# np.save("../../data/SISO_three-masses/u_test",u_test)
np.save("../../data/SISO_three-masses/u",u)
np.save("../../data/SISO_three-masses/x",x)

# %%
