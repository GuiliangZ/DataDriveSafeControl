import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from scipy.optimize import minimize
import control as ctrl

np.set_printoptions(precision=3,suppress=True)

def getA(state,dt):
    """
    Double integrator model A matrix - state
    """
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0,0,1,0],
        [0,0,0,1]
    ])
    return A

def getB(state, dt):
    """
    Double integrator model B matrix - control
    """
    B = np.array([
        [0,0],
        [0,0],
        [0.5*dt**2,0],
        [0,0.5*dt**2]
    ])
    return B
 
def double_intergrator_SS_dynamics(A, state_t_minus_1, B, control_input_t_minus_1):
    """
    State space representation of double integrator model
    """
    state_estimate_t = (A @ state_t_minus_1) + (B @ control_input_t_minus_1)
    return state_estimate_t
     
def lqr_controller(state, goal_position, Q, R, dt, max_acceleration):
    """
    Solve the continuous-time algebraic Riccati equation and compute the LQR gain.
    Parameters:
    - A: State matrix of the system
    - B: Input matrix of the system
    - Q: State cost matrix
    - R: Input cost matrix
    - actual_state_x
    - desired_state_xf
    - dt: Simulation time step
    Returns:
    - u_lqr: control input trajectory
    """
    goal = np.append(goal_position,[[0],[0]])
    goal = goal.reshape(-1,1)
    
    x_error = state - goal
    #pdb.set_trace()
    N = 100
    P = [None] * (N + 1)
    Qf = Q
    P[N] = Qf
    A = getA(state, dt)
    B = getB(state, dt)

    for i in range(N, 0, -1):
        P[i-1] = Q + A.T @ P[i] @ A - (A.T @ P[i] @ B) @ np.linalg.pinv(
            R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)      

    K = -np.linalg.pinv(R + B.T @ P[i+1] @ B) @ B.T @ P[i+1] @ A
    u_lqr = K @ x_error
    if np.any(u_lqr > max_acceleration) and np.any(u_lqr >= 0):
        u_lqr[u_lqr>max_acceleration] = max_acceleration
    elif np.any(u_lqr < max_acceleration) and np.any(u_lqr < 0):
        u_lqr[u_lqr>max_acceleration] = max_acceleration
    print(u_lqr)
    # pdb.set_trace()
    return u_lqr

def h(state, obs, robot_radius, safety_dist):
    """Computes the Control Barrier Function."""
    # Define the control barrier function
    x_obs, y_obs, r_obs = obs
    #distance
    h = (state[0,0] - x_obs)**2 + (state[1,0] - y_obs)**2 - \
        (robot_radius + r_obs + safety_dist)**2
    #distance derivative
    h_dot = 2 * (state[0,0] - x_obs) * state[2,0] + \
        2 * (state[1,0] - y_obs) * state[3,0]
    print(h)
    print(h_dot)
    #pdb.set_trace()
    return h, h_dot

def cbf_controller(state, obs, robot_radius, safety_dist, u_nominal, gamma_fcn, dt):
    """CBF-based controller for tracking objective."""
    # CBF controller
    h_k, h_k_dot = h(state, obs, robot_radius, safety_dist)
    u_cbf = -np.sign(h_k_dot) - gamma_fcn * h_k + u_nominal
    print(u_cbf)
    #pdb.set_trace()
    u_cbf = np.reshape(u_cbf,(-1,1))
    return u_cbf

'''
    # Define the objective function for the optimization
    def objective(u):
        pdb.set_trace()
        return u[0]**2 + u[1]**2  # Example: minimize control effort

    # Constraints for optimization
    constraint = lambda u: h(state, obs, robot_radius, safety_dist)[0]

    # Initial guess for control input
    u0 = np.zeros(2)  

    # Optimization to satisfy CBF
    result = minimize(objective, u0, constraints={'type': 'ineq', 'fun': constraint})

    # Extract the optimal control input
    u_cbf = result.x
    u_cbf = np.reshape(u_cbf,(-1,1))
    print(u_cbf)
    return u_cbf
'''


'''
def clf_fcn(obj, params, symbolic_state):
    x = symbolic_state
    A = zeros(4)
    A(1, 2) = 1
    A(3, 4) = 1
    B = [0 0; 1 0; 0 0; 0 1]
    Q = eye(size(A))
    R = eye(size(B,2))
    [~,P] = lqr(A,B,Q,R)
    e = x - [params.p_d(1); 0; params.p_d(2); 0]
    clf = e' * P * e

def cbf_fcn(obj, params, symbolic_state):
    x = symbolic_state
    p_o = params.p_o; # position of the obstacle.
    r_o = params.r_o; # radius of the obstacle.
    distance = (x(1) - p_o(1))^2 + (x(3) - p_o(2))^2 - r_o^2
    derivDistance = 2*(x(1)-p_o(1))*x(2) + 2*(x(3)-p_o(2))*x(4)
    cbf = derivDistance + params.cbf_gamma0 * distance
'''