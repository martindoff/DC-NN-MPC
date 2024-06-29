""" pvtol model 

(c) Martin Doff-Sotta, University of Oxford (martin.doff-sotta@eng.ox.ac.uk)

"""
import numpy as np
import cvxpy as cp

## Dynamics 
def f_full(x, u, p):
    """ PVTOL full dynamics
    
    Let (x1, x2, x3, x4) = (alpha, y', z', alpha'), the state space model is: 

    x1' = x4
    x2' = (g + u1)*sin(x1)
    x3' = (g + u1)*cos(x1) - g
    x4' = u2
    
    """
    da  = x[3]
    ddy = (u[0] + p.g)*np.sin(x[0])
    ddz = (u[0] + p.g)*np.cos(x[0]) - p.g
    dda = u[1]
    
    return np.array([da, ddy, ddz, dda])

def f(alpha, u, p):
    """ PVTOL nonlinear dynamics """
    ddy = (u + p.g)*np.sin(alpha)
    ddz = (u + p.g)*np.cos(alpha) - p.g
    
    return np.vstack([ddy, ddz])   

def ddy(alpha, u, p):
    """ PVTOL nonlinear y-axis dynamics """
    return (u + p.g)*np.sin(alpha)

def ddz(alpha, u, p):
    """ PVTOL nonlinear z-axis dynamics """
    return (u + p.g)*np.cos(alpha) - p.g

## Linearise model
def derivative_weight(x, sigma, dsigma, weights, N_state):
    """ Derivative of model from weights 
    
    N_state: number of states in x =[x_k, u_k]
    """
    
    # First layer
    x0 = x
    W = weights[0].T
    b = weights[1].T
    z = W @ x + b[:, None]
    x = sigma(z)
    
    # Derivative
    A = dsigma(z[:, 0]) @ W[:, 0:N_state]
    B = dsigma(z[:, 0]) @ W[:, N_state:]

    # Internal layers
    N = (len(weights)-4)//4
    for i in range(N):
        Wx = weights[2+i*4].T
        bx = weights[2+i*4+1].T
        W0 = weights[2+i*4+2].T
        b0 = weights[2+i*4+3].T
        
        z = Wx @ x  + bx[:, None] +  W0 @ x0 + b0[:, None]
        x = sigma(z)
        
        # Derivative
        A =  dsigma(z[:, 0]) @ (Wx @ A + W0[:, 0:N_state])
        B =  dsigma(z[:, 0]) @ (Wx @ B + W0[:, N_state:])
    
    # Last layer
    W = weights[-2].T
    b = weights[-1].T
    z = W @ x + b[:, None]
    
    # Derivative
    A =  W @ A  # no dsigma(z) @ because no activation on last layer
    B =  W @ B  # no dsigma(z) @ because no activation on last layer
    
    return A, B
    
def linearise(x_0, u_0, weights_g, weights_h, sigma, dsigma):
    """
    Form the DC linearised continuous-time model of the PVTOL around x_0, u_0 
    
    dx/dt = (A1 - A2) x + (B1 - B2) u
    
    """
    
    # Dimensions
    N_state = x_0.shape[0]
    N_input = u_0.shape[0]
    N = u_0.shape[1]
    
    # Initialisation
    A1 = np.zeros((N, N_state, N_state))
    A2 = np.zeros((N, N_state, N_state))
    B1 = np.zeros((N, N_state, N_input))
    B2 = np.zeros((N, N_state, N_input))
    x_ = np.hstack([x_0[0, None, :].T, u_0[0,  None, :].T])
    x_ = np.vstack([x_0[0, None, :], u_0[0,  None, :]])
    
    N_state_inner = 1  # number of states for nonlinear dynamics
    N_input_inner = x_.shape[0] - N_state_inner # number of input for nonlinear dynamics 
    A_g = np.zeros((N, x_.shape[0], N_state_inner))
    B_g = np.zeros((N, x_.shape[0], N_input_inner))
    A_h = np.zeros((N, x_.shape[0], N_state_inner))
    B_h = np.zeros((N, x_.shape[0], N_input_inner))
    
    for i in range(N):
        A_g[i, :, :], B_g[i, :, :] = derivative_weight(x_[:, i, None], sigma, dsigma, weights_g, N_state_inner)
        A_h[i, :, :], B_h[i, :, :] = derivative_weight(x_[:, i, None], sigma, dsigma, weights_h, N_state_inner)
                                                                                 
    # A
    A1[:, 0, 3] = 1 
    A1[:, 1, 0] = A_g[:, 0, 0]
    A1[:, 2, 0] = A_g[:, 1, 0]
    A2[:, 1, 0] = A_h[:, 0, 0]
    A2[:, 2, 0] = A_h[:, 1, 0]
    
    # B
    B1[:, 3, 1] = 1 
    B1[:, 1, 0] = B_g[:, 0, 0]
    B1[:, 2, 0] = B_g[:, 1, 0]
    B2[:, 1, 0] = B_h[:, 0, 0]
    B2[:, 2, 0] = B_h[:, 1, 0]
    
    return A1, B1, A2, B2
    
def discretise(A, B, delta):
    """
    Convert a continuous-time state space model into a discrete-time state space model
    
    Input:
    - A, B: continuous-time state space model
    - delta: time step
    
    Output: 
    - A_d, B_d: discrete-time state space model
    
    """
    # Dimensions
    N_state = A.shape[1]
    
    # Linearised discrete-time model
    A_d = np.eye(N_state) + delta*A
    B_d = delta*B
    
    return A_d, B_d
    
def linearise_true(x_0, u_0, p):
    """Form the linearised continuous-time model of the PVTOL around x_0, u_0 
    
    dx/dt = A x + B u
    

    A = [               0 0 0 1
           (g+u1)*cos(x1) 0 0 0
          -(g+u1)*sin(x1) 0 0 0
                        0 0 0 0], 
    
    B = [      0 0
         sin(x1) 0
         cos(x1) 0
               0 1], 
    
    and I is the identity.  
    
    Input: 
    - x_0: guess state trajectory
    - u_0: guess input trajectory
    - p: structure of parameters
    
    Output: 
    - A, B: continuous-time matrices"""
    
    
    
    # Dimensions
    N_state = x_0.shape[0]
    N_input = u_0.shape[0]
    N = u_0.shape[1]
    
    # Initialisation
    A = np.zeros((N, N_state, N_state))
    B = np.zeros((N, N_state, N_input))
    
    # A
    A[:, 0, 3] = 1 
    A[:, 1, 0] = (p.g + u_0[0, :])*np.cos(x_0[0, :])
    A[:, 2, 0] = -(p.g + u_0[0, :])*np.sin(x_0[0, :])
    
    # B
    B[:, 1, 0] = np.sin(x_0[0, :])
    B[:, 2, 0] = np.cos(x_0[0, :])
    B[:, 3, 1] = 1 
    
    return A, B

def feasibility(f, x_0, x_r, delta, N, param):
    """ Generate a feasible trajectory for the PVTOL """
    
    
    # Initialisation
    N_state = x_0.shape[0]
    N_input = 2
    u = np.zeros((N_input, N))                     # control input                     
    x = np.zeros((N_state, N+1))
    t = np.zeros((N+1, ))
    
    # Reference
    zr = 0
    ar = param.h_r[0]
        
    # Compute trajectory
    x[:, 0] = x_0
    h = -1  # height
    for i in range(N):
        # Feedback linearisation
        u1 = (1*(zr-h + 1*-x[2, i])+param.g)/np.cos(x[0, i]) -param.g
        u2 = 3*(ar-x[0, i]) + 5*-x[3, i]
        u[:, i] = np.array([u1, u2])
        
        # Dynamic update
        x[:, i+1] = x[:, i] + delta*(f_full(x[:, i], u[:, i], param))
        t[i+1] = t[i] + delta
        
        # height update
        h = h + delta*x[2, i]

    return x, u, t

def interp_feas(t_0, t_feas, x_feas, u_feas):
    
    N = t_0.shape[0]-1
    N_state = x_feas.shape[0]
    N_input = u_feas.shape[0]
    x_0 = np.zeros((N_state, N+1))
    u_0 = np.zeros((N_input, N))
    
    for i in range(N_state):
        x_0[i, :] = np.interp(t_0, t_feas, x_feas[i, :])
    
    for i in range(N_input):
        u_0[i, :] = np.interp(t_0[:-1], t_feas[:-1], u_feas[i, :])
        
    return x_0, u_0