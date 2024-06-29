""" Solve optimisation problem

(c) 06/2024 - Martin Doff-Sotta, University of Oxford (martin.doff-sotta@eng.ox.ac.uk)

"""
import numpy as np
from scipy.linalg import block_diag
import time
import cvxpy as cp

def cvx_opt_elem(x_p, x_0, u_0, x_r, u_r, delta, param, sqrt_Q, sqrt_R, sqrt_Q_N, gamma_N, 
                 K, A1, A2, B1, B2, W_low, W_up, g, h, g_0, h_0):
    """ 
    
    Solve optimisation with elementwise bounds and quadratic objective
    
    """
    
    # Problem dimensions
    N_state = x_0.shape[0]
    N_input = u_0.shape[0]
    N = u_0.shape[1]
    N_ver = 2**N_state                     # number of vertices 
        
    # Optimisation variables
    theta = cp.Variable(N+1)               # state cost
    chi = cp.Variable(N)                   # input cost
    v = cp.Variable((N_input, N))          # feedforward input from u = v + K x
    X_lb = cp.Variable((N_state, N+1))     # state lower bound
    X_ub = cp.Variable((N_state, N+1))     # state upper bound
    X_ = {}                                # create dictionary for 3D variable 
    for l in range(N_ver):
        X_[l] = cp.Expression

    # Define blockdiag matrices for page-wise matrix multiplication
    K_ = block_diag(*K)
    
    A1_ = block_diag(*A1)
    A2_ = block_diag(*A2)
    B1_ = block_diag(*B1)
    B2_ = block_diag(*B2)
           
    # Objective
    objective = cp.Minimize(cp.sum_squares(theta) + cp.sum_squares(chi))
        
    # Constraints
    constr = []
        
    # Assemble vertices 
    X_[0]  = X_lb                                                        # 0000
    X_[1]  = X_ub                                                        # 1111
    X_[2]  = cp.vstack([X_lb[0, :], X_lb[1, :], X_lb[2, :], X_ub[3, :]]) # 0001
    X_[3]  = cp.vstack([X_lb[0, :], X_lb[1, :], X_ub[2, :], X_lb[3, :]]) # 0010
    X_[4]  = cp.vstack([X_lb[0, :], X_lb[1, :], X_ub[2, :], X_ub[3, :]]) # 0011
    X_[5]  = cp.vstack([X_lb[0, :], X_ub[1, :], X_lb[2, :], X_lb[3, :]]) # 0100
    X_[6]  = cp.vstack([X_lb[0, :], X_ub[1, :], X_lb[2, :], X_ub[3, :]]) # 0101
    X_[7]  = cp.vstack([X_lb[0, :], X_ub[1, :], X_ub[2, :], X_lb[3, :]]) # 0110
    X_[8]  = cp.vstack([X_lb[0, :], X_ub[1, :], X_ub[2, :], X_ub[3, :]]) # 0111
    X_[9]  = cp.vstack([X_ub[0, :], X_lb[1, :], X_lb[2, :], X_lb[3, :]]) # 1000
    X_[10] = cp.vstack([X_ub[0, :], X_lb[1, :], X_lb[2, :], X_ub[3, :]]) # 1001
    X_[11] = cp.vstack([X_ub[0, :], X_lb[1, :], X_ub[2, :], X_lb[3, :]]) # 1010
    X_[12] = cp.vstack([X_ub[0, :], X_lb[1, :], X_ub[2, :], X_ub[3, :]]) # 1011
    X_[13] = cp.vstack([X_ub[0, :], X_ub[1, :], X_lb[2, :], X_lb[3, :]]) # 1100
    X_[14] = cp.vstack([X_ub[0, :], X_ub[1, :], X_lb[2, :], X_ub[3, :]]) # 1101
    X_[15] = cp.vstack([X_ub[0, :], X_ub[1, :], X_ub[2, :], X_lb[3, :]]) # 1110
      
    for l in range(N_ver):
        # Define some useful variables
        X = X_[l]
        X_r = cp.reshape(X[:, :-1], (N_state*N,1))
        s_r = cp.reshape(X[:, :-1]-x_0[:, :-1], (N_state * N, 1))
        K_x = cp.reshape(K_ @ X_r, (N_input, N))
        v_r = cp.reshape(v + K_x - u_0, (N_input*N,1))

            
        A1_s = cp.reshape(A1_ @ s_r, (A1.shape[1], N))
        A2_s = cp.reshape(A2_ @ s_r, (A2.shape[1], N))
        B1_v = cp.reshape(B1_ @ v_r, (B1.shape[1], N))
        B2_v = cp.reshape(B2_ @ v_r, (B2.shape[1], N))
            
        # Objective constraints 
        constr += [chi        >= (cp.norm(sqrt_R   @ (v + K_x - u_r)))              ]
        constr += [theta[:-1] >= (cp.norm(sqrt_Q   @ (X[:, :-1] - x_r[:, :-1])))]
        constr += [theta[-1]  >= (cp.norm(sqrt_Q_N @ (X[:,  -1] - x_r[:,  -1])))]
            
        # Input constraints  
        constr += [v + K_x  >= param.u_min[:, None],
                   v + K_x  <= param.u_max[:, None]]
            
        # Nonlinear tube constraints x3' = (g + u1)*cos(x1)-g & x2' = (g + u1)*sin(x1)
        z = cp.vstack([X[0, :-1], v[0,:] + K_x[0,:]])
        
        constr += [X_ub[1:3, 1:] >=  X[1:3, :-1] + delta*(g(z)\
             -(h_0 + A2_s[1:3, :] + B2_v[1:3, :]) + W_up)]
        
        constr += [X_lb[1:3,1:] <=  X[1:3,:-1] + delta*(g_0 + A1_s[1:3,:] + B1_v[1:3,:]\
                      - h(z) + W_low)]
        
    # Linear tube constraints
    constr += [X_ub[0, 1:] >=  X_ub[0, :-1] + delta*X_ub[3, :-1],
               X_lb[0, 1:] <=  X_lb[0, :-1] + delta*X_lb[3, :-1]]         # x1' = x4
        
    constr += [X_ub[3, 1:] >=  X_ub[3, :-1] + delta*(v[1,:] + K_x[1,:]),
               X_lb[3, 1:] <=  X_lb[3, :-1] + delta*(v[1,:] + K_x[1,:])]  # x4' = u2
                  
    # State constraints
    constr += [X_lb[:, :-1] >= param.x_min[:, None],
               X_ub[:, :-1] <= param.x_max[:, None], 
               X_lb[:, 0] == x_p, 
               X_ub[:, 0] == x_p] 
                        
    # Terminal set constraint 
    constr += [np.sqrt(gamma_N) >= theta[-1]] 

    # Solve problem
    problem = cp.Problem(objective, constr)
    problem.solve(solver = cp.MOSEK, verbose=False)
    
    return problem, X_lb, X_ub, v

def cvx_opt_elem_fast(x_p, x_0, u_0, x_r, u_r, delta, param, sqrt_Q, sqrt_R, sqrt_Q_N, 
                      gamma_N, K, A1, A2, B1, B2, W_low, W_up, g, h, g_0, h_0):
    """ 
    
    Solve optimisation with elementwise bounds and linear objective
    
    """
    
    Q_N = sqrt_Q_N@sqrt_Q_N 
    
    # Problem dimensions
    N_state = x_0.shape[0]
    N_input = u_0.shape[0]
    N = u_0.shape[1]
    N_ver = 2**N_state                     # number of vertices 
        
    # Optimisation variables
    theta = cp.Variable(N+1)               # state cost
    chi = cp.Variable(N)                   # input cost
    v = cp.Variable((N_input, N))          # feedforward input from u = v + K x
    X_lb = cp.Variable((N_state, N+1))     # state lower bound
    X_ub = cp.Variable((N_state, N+1))     # state upper bound
    X_ = {}                                # create dictionary for 3D variable 
    for l in range(N_ver):
        X_[l] = cp.Expression

    # Define blockdiag matrices for page-wise matrix multiplication
    K_ = block_diag(*K)
    
    A1_ = block_diag(*A1)
    A2_ = block_diag(*A2)
    B1_ = block_diag(*B1)
    B2_ = block_diag(*B2)
           
    # Objective
    objective = cp.Minimize(cp.sum(theta) + cp.sum(chi))
        
    # Constraints
    constr = []
        
    # Assemble vertices 
    X_[0]  = X_lb                                                        # 0000
    X_[1]  = X_ub                                                        # 1111
    X_[2]  = cp.vstack([X_lb[0, :], X_lb[1, :], X_lb[2, :], X_ub[3, :]]) # 0001
    X_[3]  = cp.vstack([X_lb[0, :], X_lb[1, :], X_ub[2, :], X_lb[3, :]]) # 0010
    X_[4]  = cp.vstack([X_lb[0, :], X_lb[1, :], X_ub[2, :], X_ub[3, :]]) # 0011
    X_[5]  = cp.vstack([X_lb[0, :], X_ub[1, :], X_lb[2, :], X_lb[3, :]]) # 0100
    X_[6]  = cp.vstack([X_lb[0, :], X_ub[1, :], X_lb[2, :], X_ub[3, :]]) # 0101
    X_[7]  = cp.vstack([X_lb[0, :], X_ub[1, :], X_ub[2, :], X_lb[3, :]]) # 0110
    X_[8]  = cp.vstack([X_lb[0, :], X_ub[1, :], X_ub[2, :], X_ub[3, :]]) # 0111
    X_[9]  = cp.vstack([X_ub[0, :], X_lb[1, :], X_lb[2, :], X_lb[3, :]]) # 1000
    X_[10] = cp.vstack([X_ub[0, :], X_lb[1, :], X_lb[2, :], X_ub[3, :]]) # 1001
    X_[11] = cp.vstack([X_ub[0, :], X_lb[1, :], X_ub[2, :], X_lb[3, :]]) # 1010
    X_[12] = cp.vstack([X_ub[0, :], X_lb[1, :], X_ub[2, :], X_ub[3, :]]) # 1011
    X_[13] = cp.vstack([X_ub[0, :], X_ub[1, :], X_lb[2, :], X_lb[3, :]]) # 1100
    X_[14] = cp.vstack([X_ub[0, :], X_ub[1, :], X_lb[2, :], X_ub[3, :]]) # 1101
    X_[15] = cp.vstack([X_ub[0, :], X_ub[1, :], X_ub[2, :], X_lb[3, :]]) # 1110
      
    for l in range(N_ver):
        # Define some useful variables
        X = X_[l]
        X_r = cp.reshape(X[:, :-1], (N_state*N,1))
        s_r = cp.reshape(X[:, :-1]-x_0[:, :-1], (N_state * N, 1))
        K_x = cp.reshape(K_ @ X_r, (N_input, N))
        v_r = cp.reshape(v + K_x - u_0, (N_input*N,1))

            
        A1_s = cp.reshape(A1_ @ s_r, (A1.shape[1], N))
        A2_s = cp.reshape(A2_ @ s_r, (A2.shape[1], N))
        B1_v = cp.reshape(B1_ @ v_r, (B1.shape[1], N))
        B2_v = cp.reshape(B2_ @ v_r, (B2.shape[1], N))
            
        # Objective constraints 
        constr += [chi      >= (cp.square(sqrt_R[0, 0]*(v[0, :] + K_x[0, :] - u_r[0, :]))\
                             + cp.square(sqrt_R[1, 1]*(v[1, :] + K_x[1, :] - u_r[1, :])))]
        constr += [theta[:-1] >= (cp.square(sqrt_Q[0, 0]*(X[0, :-1] - x_r[0, :-1]))\
                                 +cp.square(sqrt_Q[1, 1]*(X[1, :-1] - x_r[1, :-1]))\
                                 +cp.square(sqrt_Q[2, 2]*(X[2, :-1] - x_r[2, :-1]))\
                                 +cp.square(sqrt_Q[3, 3]*(X[3, :-1] - x_r[3, :-1])))]
        constr += [theta[-1]  >= cp.quad_form(X[:,  -1] - x_r[:,  -1], Q_N)]
            
        # Input constraints  
        constr += [v + K_x  >= param.u_min[:, None],
                   v + K_x  <= param.u_max[:, None]]
            
        # Nonlinear tube constraints x3' = (g + u1)*cos(x1)-g & x2' = (g + u1)*sin(x1)
        z = cp.vstack([X[0, :-1], v[0,:] + K_x[0,:]])
        
        constr += [X_ub[1:3, 1:] >=  X[1:3, :-1] + delta*(g(z)\
             -(h_0 + A2_s[1:3, :] + B2_v[1:3, :]) + W_up)]
        
        constr += [X_lb[1:3,1:] <=  X[1:3,:-1] + delta*(g_0 + A1_s[1:3,:] + B1_v[1:3,:]\
                      - h(z) + W_low)]
        
    # Linear tube constraints
    constr += [X_ub[0, 1:] >=  X_ub[0, :-1] + delta*X_ub[3, :-1],
               X_lb[0, 1:] <=  X_lb[0, :-1] + delta*X_lb[3, :-1]]         # x1' = x4
        
    constr += [X_ub[3, 1:] >=  X_ub[3, :-1] + delta*(v[1,:] + K_x[1,:]),
               X_lb[3, 1:] <=  X_lb[3, :-1] + delta*(v[1,:] + K_x[1,:])]  # x4' = u2
                  
    # State constraints
    constr += [X_lb[:, :-1] >= param.x_min[:, None],
               X_ub[:, :-1] <= param.x_max[:, None], 
               X_lb[:, 0] == x_p, 
               X_ub[:, 0] == x_p] 
                        
    # Terminal set constraint 
    constr += [np.sqrt(gamma_N) >= theta[-1]] 

    # Solve problem
    problem = cp.Problem(objective, constr)
    problem.solve(solver = cp.MOSEK, verbose=False)
    
    return problem, X_lb, X_ub, v

def cvx_opt_simplex(x_p, x_0, u_0, x_r, u_r, delta, param, sqrt_Q, sqrt_R, sqrt_Q_N, 
                    gamma_N, K, A1, A2, B1, B2, W_low, W_up, g, h, g_0, h_0):
    """ 
    
    Solve optimisation with simplex bounds and quadratic objective
    
    """
    
    # Problem dimensions
    N_state = x_0.shape[0]
    N_input = u_0.shape[0]
    N = u_0.shape[1]
    N_ver = N_state + 1                    # number of vertices 
        
    # Optimisation variables
    theta = cp.Variable(N+1)               # state cost
    chi = cp.Variable(N)                   # input cost
    v = cp.Variable((N_input, N))          # feedforward input from u = v + K x
    alpha = cp.Variable((N_state, N+1))    # Optimization variable for the simplex
    beta = cp.Variable(N+1)                # Optimization variable for the simplex
    X_ = {}                                # create dictionary for 3D variable 
    for l in range(N_ver):
        X_[l] = cp.Expression

    # Define blockdiag matrices for page-wise matrix multiplication
    K_ = block_diag(*K)
    
    A1_ = block_diag(*A1)
    A2_ = block_diag(*A2)
    B1_ = block_diag(*B1)
    B2_ = block_diag(*B2)
           
    # Objective
    objective = cp.Minimize(cp.sum_squares(theta) + cp.sum_squares(chi))
        
    # Constraints
    constr = []
        
    # Assemble vertices
    X_[0] = cp.vstack(-alpha[:, :])

    X_[1] = cp.vstack([(beta + cp.sum([alpha[j, :] for j in range(N_state) if j != 0])),
                        -alpha[1, :],
                        -alpha[2, :],
                        -alpha[3, :]])

    X_[2] = cp.vstack([-alpha[0, :],
                        (beta + cp.sum([alpha[j, :] for j in range(N_state) if j != 1])),
                        -alpha[2, :],
                        -alpha[3, :]])

    X_[3] = cp.vstack([-alpha[0, :],
                       -alpha[1, :],
                       (beta + cp.sum([alpha[j, :] for j in range(N_state) if j != 2])),
                       -alpha[3, :]])

    X_[4] = cp.vstack([-alpha[0, :],
                       -alpha[1, :],
                       -alpha[2, :],
                       (beta + cp.sum([alpha[j, :] for j in range(N_state) if j != 3])) ]) 
      
    for l in range(N_ver):
        # Define some useful variables
        X = X_[l]
        X_r = cp.reshape(X[:, :-1], (N_state*N,1))
        s_r = cp.reshape(X[:, :-1]-x_0[:, :-1], (N_state * N, 1))
        K_x = cp.reshape(K_ @ X_r, (N_input, N))
        v_r = cp.reshape(v + K_x - u_0, (N_input*N,1))

            
        A1_s = cp.reshape(A1_ @ s_r, (A1.shape[1], N))
        A2_s = cp.reshape(A2_ @ s_r, (A2.shape[1], N))
        B1_v = cp.reshape(B1_ @ v_r, (B1.shape[1], N))
        B2_v = cp.reshape(B2_ @ v_r, (B2.shape[1], N))
            
        # Objective constraints 
        constr += [chi        >= (cp.norm(sqrt_R   @ (v + K_x - u_r)))              ]
        constr += [theta[:-1] >= (cp.norm(sqrt_Q   @ (X[:, :-1] - x_r[:, :-1])))]
        constr += [theta[-1]  >= (cp.norm(sqrt_Q_N @ (X[:,  -1] - x_r[:,  -1])))]
            
        # Input constraints  
        constr += [v + K_x  >= param.u_min[:, None],
                   v + K_x  <= param.u_max[:, None]]
            
        # Nonlinear tube constraints
        z = cp.vstack([X[0, :-1], v[0,:] + K_x[0,:]])
        
        # CHECK W_up & W_low
        constr += [beta[1:] >=  cp.sum(X[:, :-1], axis=0) + delta*(X[3, :-1]\
        + cp.sum(g(z) - h_0, axis=0) -( A2_s[1, :] + B2_v[1, :])\
         -( A2_s[2, :] + B2_v[2, :]) + W_up\
        + v[1,:] + K_x[1,:])]
        
        constr += [-alpha[1:3,1:] <=  X[1:3,:-1] + delta*(g_0 + A1_s[1:3,:] + B1_v[1:3,:]\
                      - h(z) + W_low)] # x3' = (g + u1)*cos(x1)-g & x2' = (g + u1)*sin(x1)
                      
        # Linear tube constraints
        constr += [-alpha[0, 1:] <=  X[0, :-1] + delta*X[3, :-1]] # x1' = x4
        
        constr += [-alpha[3, 1:] <=  X[3, :-1] + delta*(v[1,:] + K_x[1,:])] # x4' = u2
                  
        # State constraints
        constr += [X[:, :-1] >= param.x_min[:, None],
                   X[:, :-1] <= param.x_max[:, None]]
                   
        # Initial state
        #constr += [-x_p <= alpha[:,0], cp.sum(x_p, axis=0) <= beta[0]]
        #constr += [beta[0] == cp.sum(alpha[:,0]), -alpha[:,0] == x_p]
        #constr += [X[:, 0] == x_p]
                        
    # Terminal set constraint 
    constr += [np.sqrt(gamma_N) >= theta[-1]] 

    # Solve problem
    problem = cp.Problem(objective, constr)
    problem.solve(solver = cp.MOSEK, verbose=False)
    
    return problem, alpha, beta, v

def cvx_opt_simplex_fast(x_p, x_0, u_0, x_r, u_r, delta, param, sqrt_Q, sqrt_R, sqrt_Q_N, 
                    gamma_N, K, A1, A2, B1, B2, W_low, W_up, g, h, g_0, h_0):
    """ 
    
    Solve optimisation with simplex bounds and linear objective
    
    """
    
    Q_N = sqrt_Q_N@sqrt_Q_N 
    
    # Problem dimensions
    N_state = x_0.shape[0]
    N_input = u_0.shape[0]
    N = u_0.shape[1]
    N_ver = N_state + 1                    # number of vertices 
        
    # Optimisation variables
    theta = cp.Variable(N+1)               # state cost
    chi = cp.Variable(N)                   # input cost
    v = cp.Variable((N_input, N))          # feedforward input from u = v + K x
    alpha = cp.Variable((N_state, N+1))    # Optimization variable for the simplex
    beta = cp.Variable(N+1)                # Optimization variable for the simplex
    X_ = {}                                # create dictionary for 3D variable 
    for l in range(N_ver):
        X_[l] = cp.Expression

    # Define blockdiag matrices for page-wise matrix multiplication
    K_ = block_diag(*K)
    
    A1_ = block_diag(*A1)
    A2_ = block_diag(*A2)
    B1_ = block_diag(*B1)
    B2_ = block_diag(*B2)
           
    # Objective
    objective = cp.Minimize(cp.sum(theta) + cp.sum(chi))
        
    # Constraints
    constr = []
        
    # Assemble vertices
    X_[0] = cp.vstack(-alpha[:, :])

    X_[1] = cp.vstack([(beta + cp.sum([alpha[j, :] for j in range(N_state) if j != 0])),
                        -alpha[1, :],
                        -alpha[2, :],
                        -alpha[3, :]])

    X_[2] = cp.vstack([-alpha[0, :],
                        (beta + cp.sum([alpha[j, :] for j in range(N_state) if j != 1])),
                        -alpha[2, :],
                        -alpha[3, :]])

    X_[3] = cp.vstack([-alpha[0, :],
                       -alpha[1, :],
                       (beta + cp.sum([alpha[j, :] for j in range(N_state) if j != 2])),
                       -alpha[3, :]])

    X_[4] = cp.vstack([-alpha[0, :],
                       -alpha[1, :],
                       -alpha[2, :],
                       (beta + cp.sum([alpha[j, :] for j in range(N_state) if j != 3])) ]) 
      
    for l in range(N_ver):
        # Define some useful variables
        X = X_[l]
        X_r = cp.reshape(X[:, :-1], (N_state*N,1))
        s_r = cp.reshape(X[:, :-1]-x_0[:, :-1], (N_state * N, 1))
        K_x = cp.reshape(K_ @ X_r, (N_input, N))
        v_r = cp.reshape(v + K_x - u_0, (N_input*N,1))

            
        A1_s = cp.reshape(A1_ @ s_r, (A1.shape[1], N))
        A2_s = cp.reshape(A2_ @ s_r, (A2.shape[1], N))
        B1_v = cp.reshape(B1_ @ v_r, (B1.shape[1], N))
        B2_v = cp.reshape(B2_ @ v_r, (B2.shape[1], N))
            
        # Objective constraints 
        constr += [chi      >= (cp.square(sqrt_R[0, 0]*(v[0, :] + K_x[0, :] - u_r[0, :]))\
                             + cp.square(sqrt_R[1, 1]*(v[1, :] + K_x[1, :] - u_r[1, :])))]
        constr += [theta[:-1] >= (cp.square(sqrt_Q[0, 0]*(X[0, :-1] - x_r[0, :-1]))\
                                 +cp.square(sqrt_Q[1, 1]*(X[1, :-1] - x_r[1, :-1]))\
                                 +cp.square(sqrt_Q[2, 2]*(X[2, :-1] - x_r[2, :-1]))\
                                 +cp.square(sqrt_Q[3, 3]*(X[3, :-1] - x_r[3, :-1])))]
        constr += [theta[-1]  >= cp.quad_form(X[:,  -1] - x_r[:,  -1], Q_N)]
            
        # Input constraints  
        constr += [v + K_x  >= param.u_min[:, None],
                   v + K_x  <= param.u_max[:, None]]
            
        # Nonlinear tube constraints
        z = cp.vstack([X[0, :-1], v[0,:] + K_x[0,:]])
        
        # CHECK W_up & W_low
        constr += [beta[1:] >=  cp.sum(X[:, :-1], axis=0) + delta*(X[3, :-1]\
        + cp.sum(g(z) - h_0, axis=0) -( A2_s[1, :] + B2_v[1, :])\
         -( A2_s[2, :] + B2_v[2, :]) + W_up\
        + v[1,:] + K_x[1,:])]
        
        constr += [-alpha[1:3,1:] <=  X[1:3,:-1] + delta*(g_0 + A1_s[1:3,:] + B1_v[1:3,:]\
                      - h(z) + W_low)] # x3' = (g + u1)*cos(x1)-g & x2' = (g + u1)*sin(x1)
                      
        # Linear tube constraints
        constr += [-alpha[0, 1:] <=  X[0, :-1] + delta*X[3, :-1]] # x1' = x4
        
        constr += [-alpha[3, 1:] <=  X[3, :-1] + delta*(v[1,:] + K_x[1,:])] # x4' = u2
                  
        # State constraints
        constr += [X[:, :-1] >= param.x_min[:, None],
                   X[:, :-1] <= param.x_max[:, None]]
                   
        # Initial state
        constr += [-x_p <= alpha[:,0], cp.sum(x_p, axis=0) <= beta[0]]
        #constr += [beta[0] == cp.sum(alpha[:,0]), -alpha[:,0] == x_p]
        #constr += [X[:, 0] == x_p]
                        
    # Terminal set constraint 
    constr += [np.sqrt(gamma_N) >= theta[-1]] 

    # Solve problem
    problem = cp.Problem(objective, constr)
    problem.solve(solver = cp.MOSEK, verbose=False)
    
    return problem, alpha, beta, v