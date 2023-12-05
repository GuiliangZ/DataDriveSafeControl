import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from scipy.optimize import minimize
import control as ctrl

class Vehicle_Dynamics:
    def __init__(self, state, u, dt):
        self.state = state
        self.u = u 
        self.dt = dt


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
    
    def dubins_car_dynamics(state, u, v, dt):
        '''
        Update system state to next dt step
        state: np.array with [x1;x2;theta]
        control: u is acceleration [theta_dot]
        v: constant velocity
        '''
        state_new = np.empty((3,1))
        state_new[0,0] = state[0,0] + v * np.cos(state[2,0]) * dt   # x position
        state_new[1,0] = state[1,0] + v * np.sin(state[2,0]) * dt   # y position
        state_new[2,0] = state[2,0] + u * dt    # theta
        return state_new

    def unicycle_3(state, u, dt):
        '''
        Update system state to next dt step
        state: np.array with [x1;x2;theta]
        control: u is acceleration [v1;theta_dot]
        '''
        state_new = np.empty((3,1))
        state_new[0,0] = state[0,0] + u[0,0] * np.cos(state[2,0]) * dt  # x position
        state_new[1,0] = state[1,0] + u[0,0] * np.sin(state[2,0]) * dt  # y position
        state_new[2,0] = state[2,0] + u[1,0] * dt   #theta - velocity is not constant but is a control u input
        return state_new


    def unicycle_4(state, u, dt):
        '''
        Update system state to next dt step
        state: np.array with [x1;x2;v1;theta]
        control: u is acceleration [a1;theta_dot]
        '''
        state_new = np.empty((4,1))
        state_new[0,0] = state[0,0] + state[2,0] * np.cos(state[3,0]) * dt  #x position
        state_new[1,0] = state[1,0] + state[2,0] * np.sin(state[3,0]) * dt  #y position
        state_new[2,0] = state[2,0] + u[0,0] * dt # velocity
        state_new[3,0] = state[3,0] + u[1,0] * dt # theta
        return state_new

