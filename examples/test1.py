import numpy as np
import matplotlib.pyplot as plt

# Define the dynamics of the system (for demonstration purposes)
def dynamics(x, u):
    # Placeholder dynamics; replace with your own system dynamics
    A = np.array([[1, 0.1], [0, 1]])
    B = np.array([[0], [0.1]])
    return A.dot(x) + B.dot(u)

# Define the safety constraint
def safety_constraint(x):
    # Placeholder safety constraint; replace with your own constraint
    return x[0] + x[1] - 1

# Define the control barrier function (CBF)
def control_barrier_function(x):
    # Placeholder CBF; replace with your own function
    return x[0] + x[1] - 1

# Define the control law using the CBF
def control_law(x, lambda_cbf):
    # Placeholder control law; replace with your own control law
    u = -lambda_cbf * control_barrier_function(x)
    return u

# Simulation parameters
dt = 0.1
T = 5
num_steps = int(T / dt)
lambda_cbf = 1.0

# Initial condition
x = np.array([0.5, 0.5])

# Simulation loop
for t in range(num_steps):
    # Evaluate safety constraint and CBF
    h = safety_constraint(x)
    cbf = control_barrier_function(x)

    # Check safety constraint satisfaction
    # if h <= 0:
    #     print(f"Unsafe state at t={t*dt}: h(x) = {h}")

    # Apply control law
    u = control_law(x, lambda_cbf)

    # Integrate dynamics
    x = x + dynamics(x, u) * dt

# Plot the trajectory
plt.plot(x[0], x[1], 'ro', label='Final State')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('CBF-Based Control Simulation')
plt.show()

