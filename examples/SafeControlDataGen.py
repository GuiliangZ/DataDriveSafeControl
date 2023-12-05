import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def control_barrier_function(x):
    # Define the control barrier function
    return x[0] - 1

def system_dynamics(t, x, u):
    # Define the system dynamics (example: simple linear system)
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    x_dot = A.dot(x) + B.dot(u)
    return x_dot

def cbf_controller(t, x):
    # CBF controller
    u = -np.sign(control_barrier_function(x))
    return u

# Initial conditions
x0 = np.array([0, 0])

# Time span
t_span = (0, 5)

# Solve the system using the CBF controller
sol = solve_ivp(
    fun=lambda t, x: system_dynamics(t, x, cbf_controller(t, x)),
    t_span=t_span,
    y0=x0,
    method='RK45',
    dense_output=True
)

# Plot the results
plt.plot(sol.t, sol.y[0], label='Position')
plt.plot(sol.t, sol.y[1], label='Velocity')
plt.xlabel('Time')
plt.ylabel('State')
plt.legend()
plt.show()







