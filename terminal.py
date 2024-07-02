""" Terminal set computation

(c) Martin Doff-Sotta, University of Oxford (martin.doff-sotta@eng.ox.ac.uk)

"""
import numpy as np
import cvxpy as cp
from pvtol_model import linearise, discretise
import scipy.linalg 

## Cartesian product
def cartesian_product(*arrays):
    """
    Generate the n-ary cartesian product of the n input arrays
    
    Input: unpacked input arrays
    Output: matrix whose rows are ordered n-tuples formed from the input 
    """
    
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)
       
def get_terminal(param, delta, weights_g, weights_h, sigma, dsigma):
    """ Compute terminal cost, terminal constraint bound and terminal matrix
    Ouput: terminal matrix Q_N, terminal constraint bound gamma_N and terminal gain K_N 
    """
    
    # Initialisation 
    alpha = 10                                                  # objective weight
    n = len(param.x_init)                                       # number of states
    m = len(param.u_init)                                       # number of inputs
    I_s = np.eye(n)
    I_i = np.eye(m)
    Q = param.Q
    R = param.R
    C = Q[-1, None, :]
    Q_inv = np.linalg.inv(Q); 
    eps = np.finfo(float).eps
    
    
    # Variables definition (SDP)
    Q_N = cp.Variable((n,n), symmetric=True)
    S = cp.Variable((n,n), symmetric=True)
    Y = cp.Variable((m,n))
    gamma_inv = cp.Variable((1,1))
    
    # Terminal set definition 
    all_vertices = np.vstack([param.x_term[:, None], param.u_term[:, None]])  # stack state and input terminal bounds
    vertices_list = np.hstack([all_vertices, -all_vertices])  # assemble vertices
    Ver_trans = cartesian_product(*vertices_list)  # generate all possible vertices
    Ver = Ver_trans.T + np.vstack([param.h_r[:, None], param.u_r[:, None]])  # vertices of terminal set
    # Objective 
    objective = cp.Minimize(cp.trace(Q_N) + alpha*gamma_inv)
    
    # Initialise constraints
    constr = []
    
    # Constraint S = Q_N^-1
    constr += [cp.vstack([cp.hstack([S, np.eye(n)]), cp.hstack([np.eye(n), Q_N])]) >> eps*np.eye(n*2)]
    
    # Terminal cost constraint
    #Y_ = cp.vstack([np.zeros((n-m, n)), Y])
    Y_ = Y
    R_inv = np.linalg.inv(R)
    #R_ = scipy.linalg.block_diag(np.zeros((n-m, n-m)), R_inv)
    R_ = R_inv
    #R_ = cp.diag(np.array([*(0,)*(n-m), ], dtype=object))
    #CS = cp.vstack([C @ S, np.zeros_like(C)])
    O = np.zeros((n, n))
    for i in range(Ver.shape[1]):
        A1, B1, A2, B2 = linearise(Ver[0:n, i, None], 
                         Ver[n:n+m, i, None], weights_g, weights_h, sigma, dsigma)
        
        A, B = discretise(A1[0]-A2[0], B1[0]-B2[0], delta)  # discrete-time ss
        
        M = (A @ S + B @ Y)
        constr += [cp.vstack([cp.hstack([S, M.T, S, Y.T]), 
                              cp.hstack([M, S, np.zeros((n, n)), np.zeros((n, m))]),
                              cp.hstack([S, np.zeros((n, n)), Q_inv, np.zeros((n, m))]), 
                              cp.hstack([Y, np.zeros((m, n)), np.zeros((m, n)), R_inv])])\
                               >> eps*np.eye(3*n+m) ]
    
    # Terminal constraint F x + G u <= h
    G = cp.vstack([np.zeros((n, m)), np.zeros((n, m)), np.eye(m), -np.eye(m)])
    F = cp.vstack([np.eye(n), -np.eye(n), np.zeros((m, n)), np.zeros((m, n))])
    h = cp.vstack([param.x_max[:, None], -param.x_min[:, None],
                   param.u_max[:, None], -param.u_min[:, None]])
    h_loc = cp.vstack([h- (F @ param.h_r[:, None] + G @ param.u_r[:, None]), 
                       param.u_term[:, None], param.x_term[:, None]])
    G_loc = cp.vstack([G, np.eye(m), np.zeros((n, m))])
    F_loc = cp.vstack([F, np.zeros((m, n)), np.eye(n)])
    
    # Could try to uncomment this (and comment two next for loops instead)
    """for o in range(h_loc.shape[0]):
        block1 = cp.hstack([gamma_inv @ (h_loc[o,:,None])**2, 
                            F_loc[o, None, :] @ S + G_loc[o, None, :] @ Y])
        block2 = cp.hstack([(F_loc[o, None, :] @ S + G_loc[o, None, :] @ Y).T, S])
        constr += [cp.vstack([block1, block2]) >> eps*np.eye(n+1)]"""
    
    for o in range(n):
        constr += [gamma_inv * param.x_term[o]**2 - S[o, o] >> eps]
    
    for o in range(m):
        block1 = cp.hstack([gamma_inv @ param.u_term[o, None, None]**2, 
                            Y[o, None, :]])
        block2 = cp.hstack([Y[o, None, :].T, 
                            S])
        constr += [cp.vstack([block1, block2]) >> eps*np.eye(n+1)]
        
    # Solve SDP problem    
    problem = cp.Problem(objective, constr)
    problem.solve(verbose=False)
    
    # Post-processing 
    gamma_N = 1/(gamma_inv.value[0, 0])
    K_N = Y.value @ Q_N.value
    
    return Q_N.value, gamma_N, K_N

def get_terminal_(param, delta, weights_g, weights_h, sigma, dsigma):
    """ Compute terminal cost, terminal constraint bound and terminal matrix with 
    disturbance.
    Input: parameter structure param, time step delta
    Ouput: terminal matrix Q_N, terminal constraint bound gamma_N and terminal gain K_N 
    
    From: Mark Cannon March 2023
    """
    
    # Initialisation 
    n = len(param.x_init)                                       # number of states
    m = len(param.u_init)                                       # number of inputs
    I_s = np.eye(n)
    I_i = np.eye(m)
    Q = param.Q
    R = param.R
    C = Q[-1, None, :]
    Q_inv = np.linalg.inv(Q); 
    eps = np.finfo(float).eps
    
    
    # Variables definition (SDP)
    #Q_N = cp.Variable((n,n), symmetric=True)
    S = cp.Variable((n,n), symmetric=True)
    Y = cp.Variable((m,n))
    beta = cp.Variable((1,1), pos=True)
    
    # Terminal set definition 
    all_vertices = np.vstack([param.x_term[:, None], param.u_term[:, None]])  # stack state and input terminal bounds
    vertices_list = np.hstack([all_vertices, -all_vertices])  # assemble vertices
    Ver_trans = cartesian_product(*vertices_list)  # generate all possible vertices
    Ver = Ver_trans.T + np.vstack([param.h_r[:, None], param.u_r[:, None]])  # vertices of terminal set
    
    # Disturbance vertices (TO BE CHECKED)
    pre_w = np.hstack([param.W_up[:, None], param.W_low[:, None]])
    Ver_w_trans = cartesian_product(*pre_w)
    Ver_w = Ver_w_trans.T
    
    """Ver_w = np.hstack([np.vstack([param.W_low, param.W_up]), 
                       np.vstack([param.W_up,  param.W_low]), 
                       np.vstack([param.W_low, param.W_low]),
                       np.vstack([param.W_up,  param.W_up])])"""
    
    ## Problem 1
    # Objective 
    objective = cp.Minimize(beta)
    
    # Initialise constraints
    constr = []
    
    # Constraint S = Q_N^-1
    #constr += [cp.vstack([cp.hstack([S, np.eye(n)]), cp.hstack([np.eye(n), Q_N])]) >> eps*np.eye(n*2)]
    
    # Terminal cost constraint
    R_inv = np.linalg.inv(R)
    for i in range(Ver.shape[1]):
        A1, B1, A2, B2 = linearise(Ver[0:n, i, None], Ver[n:n+m, i, None], 
                                   weights_g, weights_h, sigma, dsigma)
        
        A, B = discretise(A1[0]-A2[0], B1[0]-B2[0], delta)  # discrete-time ss
        
        M = (A @ S + B @ Y)
               
        for j in range(Ver_w.shape[1]):
            W = np.vstack([0, Ver_w[:, j, None], 0])  # CHECK
            
            constr += [cp.vstack([cp.hstack([S, np.zeros((n, 1)), M.T, S, Y.T]),
                              cp.hstack([np.zeros((1, n)), beta, W.T, np.zeros((1, n)), np.zeros((1, m))]),
                              cp.hstack([M, W, S, np.zeros((n, n)), np.zeros((n, m))]),
                              cp.hstack([S, np.zeros((n, 1)), np.zeros((n, n)), Q_inv, np.zeros((n, m))]), 
                              cp.hstack([Y, np.zeros((m, 1)), np.zeros((m, n)), np.zeros((m, n)), R_inv])])\
                               >> eps*np.eye(3*n+m+1) ]
    
    # Solve SDP problem    
    problem = cp.Problem(objective, constr)
    problem.solve(verbose=False)
    
    # Post-processing 
    beta_N = beta.value[0, 0]
    Q_N = np.linalg.inv(S.value)
    K_N = Y.value @ Q_N
    
    ## Problem 2
    gamma = cp.Variable((1,1))
    
    # Objective 
    objective = cp.Maximize(gamma)
    
    # Initialise constraints
    constr = []
    
    constr += [ gamma * (Q + K_N.T @ R @ K_N) >> beta_N * Q_N]
    
    for o in range(n):
        constr += [gamma <= param.x_term[o]**2/S.value[o, o]]
    
    for o in range(m):
        constr += [gamma <= param.u_term[o]**2/( K_N[o, None, :] @ S.value @ K_N[o, None, :].T)]
        
    # Solve SDP problem    
    problem = cp.Problem(objective, constr)
    problem.solve(verbose=False)
    
    # Post-processing 
    gamma_N = gamma.value[0, 0]
    
    return Q_N, gamma_N, K_N