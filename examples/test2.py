import numpy as np
import matplotlib.pyplot as plt



def double_integrator(x, u, dt):
    n = len(x) // 2
    xnew = np.zeros_like(x)
    xnew[:n] = x[:n] + x[n:2*n] * dt + u * dt**2 / 2
    xnew[n:2*n] = x[n:2*n] + u * dt
    return xnew

def ur(x):
    return kp * (goal - x[0:2]) - kd * x[2:4]


def dr(x, ur):
    return x[0:2] + dt * x[2:4] - obstacle


def ddotr(x, ur):
    return (x[0] + dt * x[2] - obstacle[0]) * (x[2] + dt * ur[0]) + (
        x[1] + dt * x[3] - obstacle[1]
    ) * (x[3] + dt * ur[1])


kphi = 0.7


def varphi(x, ur):
    return dmin - np.linalg.norm(dr(x, ur)) - kphi * ddotr(x, ur)


def c(x, ur):
    return max(0, varphi(x, ur)) / np.linalg.norm(dr(x, ur)) / dt / kphi


def u(x):
    return ur(x) + c(x, ur(x)) * dr(x, ur(x))


# Define initial conditions and parameters
x0 = np.array([0, 0, 0])
xlist = [x0]
obstacle = np.array([0, 0])
dmin = 1
kmax = 100
dt = 0.1
thres = 0.1
goal = np.array([5, 5])

# For plotting
plt.figure(1)
plt.clf()
plt.axis([-10, 10, -10, 10])
obstacle_circle = plt.plot(
    obstacle[0] + dmin * np.cos(np.arange(0, 2 * np.pi, 0.1)),
    obstacle[1] + dmin * np.sin(np.arange(0, 2 * np.pi, 0.1)),
    '--',
)
robot_handle, = plt.plot(x0[0], x0[1], 'o', linewidth=3, color='r', markersize=20)
traj_handle, = plt.plot(xlist[0], xlist[1], 'k')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')


# Set up controller
kp = 1
kd = 2

# Simulation
for k in range(kmax):
    x = x + dt * np.array([x[2], x[3], u(x)[0], u(x)[1]])
    xlist.append(x.copy())
    robot_handle.set_xdata(x[0])
    robot_handle.set_ydata(x[1])
    plt.axis([-10, 10, -10, 10])
    if np.linalg.norm(x[0:2] - goal) < thres:
        break
    plt.pause(dt)

traj_handle.set_xdata(np.array(xlist)[:, 0])
traj_handle.set_ydata(np.array(xlist)[:, 1])
plt.title("Double Integrator with collision avoidance")
plt.legend(["Obstacle", "Robot Position", "Trajectory"])
plt.grid(True)
plt.show()

# learned model
weights = np.random.rand(4, 4)  # Replace with actual weights
xnew = lambda k, x: np.dot(weights, feature(k, x))
x = x0
xpredict = [x.copy()]

for k in range(kmax):
    x = xnew(k, x)
    xpredict.append(x.copy())

plt.figure(2)
plt.plot(np.array(xpredict).T, '--')
plt.legend(["px true", "py true", "px predict", "py predict"])
plt.xlabel("Time step")
plt.ylabel("Position")
plt.title("Learned Model Prediction vs True Dynamics")
plt.grid(True)
plt.show()