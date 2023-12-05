'''
Let's consider a simple example of a 2D point mass with 
   position x and control input u. 
We want to design a CBF-based safe controller to ensure that
   the point mass stays within a safe set defined by a circular region.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# System dynamics: x_dot = f(x, u)
def system_dynamics(t, x, u):
    # Example dynamics of a 2D point mass
    m = 1.0  # mass
    A = np.array([[0, 1], [0, 0]])
    B = np.array([0, 1 / m])
    x_dot = A @ x + B * u
    return x_dot

# Control Barrier Function (CBF) and its time derivative
def control_barrier_function(x):
    # Example CBF for a circular safe set
    r_safe = 1.0  # radius of the safe set
    return x[0]**2 + x[1]**2 - r_safe**2

def control_barrier_function_derivative(x):
    # Derivative of the CBF with respect to state
    return 2 * np.array([x[0], x[1]])

# CBF controller
def cbf_controller(x, u):
    # CBF parameters
    h = control_barrier_function(x)
    h_dot = control_barrier_function_derivative(x) @ system_dynamics(0, x, u)

    # Controller parameters
    k = 1.0  # controller gain

    # Control input
    u_cbf = u - k * h_dot / h

    return u_cbf

# Simulation
def simulate_system():
    # Initial conditions
    x0 = np.array([0.5, 0.5])

    # Time span
    t_span = (0, 10)

    # Control input
    u = 1.0

    # Solve the system with the CBF controller
    sol = solve_ivp(
        lambda t, x: system_dynamics(t, x, cbf_controller(x, u)),
        t_span,
        x0,
        method="RK45",
        dense_output=True,
    )

    return sol

# Visualization
def visualize_solution(sol):
    t_eval = np.linspace(sol.t[0], sol.t[-1], 500)
    x_eval = sol.sol(t_eval)

    plt.figure(figsize=(8, 6))
    plt.plot(x_eval[0], x_eval[1], label="Trajectory")
    plt.title("CBF-Based Safe Controller")
    plt.xlabel("Position X")
    plt.ylabel("Position Y")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main
if __name__ == "__main__":
    solution = simulate_system()
    visualize_solution(solution)
