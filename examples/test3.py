import numpy as np
import matplotlib.pyplot as plt

# Double integrator dynamics
def double_integrator(x, u, dt):
    n = len(x) // 2
    xnew = np.zeros_like(x)
    xnew[:n] = x[:n] + x[n:2*n] * dt + u * dt**2 / 2
    xnew[n:2*n] = x[n:2*n] + u * dt
    return xnew

# Set initial conditions
x0 = np.array([0, 0, 0, 0])
x = x0
xlist = [x]

# Set up obstacle
obstacle = np.array([0, 0])
dmin = 2.0  # Adjust this value according to your needs

# For plotting
plt.figure(1)
plt.clf()
plt.axis([-10, 10, -10, 10])
plt.plot(obstacle[0] + dmin * np.cos(np.arange(0, 2 * np.pi, 0.1)),
         obstacle[1] + dmin * np.sin(np.arange(0, 2 * np.pi, 0.1)), '--')
robot_handle, = plt.plot(x[0], x[1], 'o', linewidth=3, color='r', markersize=20)
traj_handle, = plt.plot(xlist[0], xlist[1], 'k')

# Set up controller
kp = 1
kd = 2
goal = np.array([10, 0])  # Adjust the goal position
ur = lambda x: kp * (goal - x[:2]) - kd * x[2:]  # Nominal control
dt = 0.1  # Time step
kmax = 100  # Maximum number of iterations
thres = 0.1  # Threshold for reaching the goal

# Simulation
for k in range(kmax):
    x = double_integrator(x, ur(x), dt)
    robot_handle.set_data(x[0], x[1])
    plt.axis([-10, 10, -10, 10])
    if np.linalg.norm(x[:2] - goal) < thres:
        break
    plt.pause(dt)
    xlist = np.column_stack((xlist, x))

traj_handle.set_data(xlist[0, :], xlist[1, :])
plt.box(on=True)
plt.title("Double Integrator with collision avoidance")

# Learned model
weights = np.random.rand()  # Replace this with your actual weights
feature = lambda k, x: np.random.rand()  # Replace this with your actual feature function

xnew = lambda k, x: np.dot(weights, feature(k, x))
x = x0
xpredict = np.array([x])

for k in range(kmax):
    x = xnew(k, x)
    xpredict = np.column_stack((xpredict, x))

plt.plot(xpredict.T, '--')
plt.legend(["px true", "py true", "px predict", "py predict"])
plt.show()