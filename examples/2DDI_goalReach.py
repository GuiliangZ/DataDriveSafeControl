import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy.integrate import solve_ivp
import sys
sys.path.append('/Users/zheng/Desktop/researchCTRL-Leung/Code/DataDriveSafeControl')
import DDSafe.DDSafe_fcn as DDS


if __name__ == "__main__":
    # Simulation parameters
    t_initial = 0
    t_end = 10  # Time span (s)
    dt = 0.1  # Time step
    dist_threshold = 0.01
    Kp = np.array([[2, 0], [0, 1]])
    Kd = np.array([[2, 0], [0, 1]])
    state = np.array([[0],[0],[0],[0]])
    goal_position = np.array([[5],[5]])
    t_range = np.arange(0, t_end + dt, dt)
    n_states = len(t_range)
    state_list = []
    u_list = []

    for i,t in enumerate(t_range):
        state_list.append(state.tolist())
        u = DDS.PD_controller(Kp, Kd, state, goal_position)
        state = DDS.double_integrator_dynamics(state, u, dt)
        u_list.append(u.tolist())
        if DDS.euclidean_distance(state[0:2],goal_position) < dist_threshold:
            break
        #import pdb; pdb.set_trace()
    # print(u_list)
    # print(state_list)
    # state_list = list(state_list)
    state_array = np.array(state_list)

#Plotting!
    print("start!")
    #Plot goal
    goal_x, goal_y = DDS.plot_goal(goal_position, radius=0.2 , obs=None)
    plt.plot(goal_x, goal_y, label = 'goal', color = 'red')
    plt.scatter(goal_position[0], goal_position[1], color='red', marker='x', label='Goal Position')
    # Plot the trajectory  
    #plt.plot(state_array.reshape(len(state_array),4)[:,0], state_array.reshape(len(state_array),4)[:,1],label='Trajectory')  
    trajectory_line, = plt.plot([], [], label='Trajectory')
    plt.title('Vehicle Trajectory')
    plt.xlabel('PositionX')
    plt.ylabel('PositionY')
    plt.legend()
    plt.xlim(0,6)
    plt.ylim(0,6)
    plt.grid(True)


    for i in range(len(state_array)):
        trajectory_line.set_xdata(state_array.reshape(len(state_array),4)[:i,0])
        trajectory_line.set_ydata(state_array.reshape(len(state_array),4)[:i,1])
        plt.pause(0.1)
    plt.pause(2)
    plt.show()
