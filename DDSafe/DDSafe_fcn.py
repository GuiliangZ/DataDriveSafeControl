import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
import control as ctrl

def single_integrator_dynamics(state, u, dt):
    '''
    state: [x1; x2]
    control u: [v1; v2]
    '''
    state_new = state + u * dt
    return state_new

def double_integrator_dynamics(state, u, dt):
    '''
    Update system state to next dt step
    state: np.array with [x1;x2;v1;v2]
    control: u is acceleration [a1;a2]
    '''
    # Double integrator dynamics
    state_new = np.empty((4,1))
    state_new[0,0] = state[0,0]+state[2,0]*dt+0.5*u[0,0]*dt**2
    state_new[1,0] = state[1,0]+state[3,0]*dt+0.5*u[1,0]*dt**2
    state_new[2,0] = state[2,0]+u[0,0]*dt
    state_new[3,0] = state[3,0]+u[1,0]*dt
    return state_new

def PD_controller(Kp, Kd, state, goal):
    '''
    Kp: proportional control 
    Kd: derivative control
    state: [x1; x2; v1; v2]
    Goal: position [x1;x2]
    '''
    # pdb.set_trace()
    # print(state[0:2,0])
    # print(state[2:4,0])
    control_input = np.dot(Kp,(goal - state[0:2])) - np.dot(Kd,state[2:4])
    return control_input

def plot_goal(center, radius, obs):
    # Create an array of theta values for the circle

    theta = np.linspace(0, 2 * np.pi, 100)
    # Parametric equations for a circle
    if obs != None:
        x_obs, y_obs, r_obs = obs
        goal_x = x_obs + r_obs * np.cos(theta)
        goal_y = y_obs + r_obs * np.sin(theta)
    else:
        goal_x = center[0] + radius * np.cos(theta)
        goal_y = center[1] + radius * np.sin(theta)
    return goal_x, goal_y

def euclidean_distance(x1, x2):
    '''
    Calculate the euclidean distance between two datapoints
    Input: two data points (1 dimensional numpy array and 2 dimensional numpy array)
    output: scalar value as the distance between those two data points. 
    '''
    return np.sqrt(np.sum((x1-x2)**2))




