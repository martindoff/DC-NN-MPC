""" Computationally tractable nonlinear robust MPC via DC programming  

This program computes a tube-based MPC control law for a PVTOL aircraft according to the 
DC-TMPC algorithm in the paper: 'Computationally tractable nonlinear robust MPC via 
DC programming' by Martin Doff-Sotta and Mark Cannon. 

The nonlinear dynamics of the aircraft is expressed as a difference of convex functions 
using neural network techniques. The algorithm is based on successive linearisations
and exploits convexity to derive tight bounds on the uncertain dynamics. 
A computationally efficient tube-based MPC algorithm is proposed to robustly stabilise the
system while guaranteeing exact satisfaction of the nonlinear dynamical constraints.  

(c) 06/2024 - Martin Doff-Sotta, University of Oxford (martin.doff-sotta@eng.ox.ac.uk)

"""

import math
import os
import sys
import time

import cvxpy as cp
import numpy as np
from scipy.linalg import sqrtm

try:
    from keras.src.layers import ReLU
except ImportError:
    from keras.layers import ReLU

import DC_decomposition as DC
import param_init as param
import param_init_DC as param_DC
from control_custom import eul, dp, seed_cost
from pvtol_model import f, linearise, discretise, feasibility, f_full, \
    interp_feas
from terminal import get_terminal as term

import matplotlib
import matplotlib.pyplot as plt

##########################################################################################
#################################### Initialisation ######################################
##########################################################################################

# Solver parameters
N = 30                                         # horizon 
T = 15                                         # terminal time
delta = T/N                                    # time step
tol1 = 10e-3                                   # tolerance   
maxIter = 1                                    # max number of iterations
N_unit = 8                                     # number of units of neural network (NN)
N_layer = 1                                    # number of hidden layers of NN
batch_size = 32                                # NN training batch size
epochs = 100                                   # NN training epochs
activation = 'relu'                            # activation function ('relu' only)
N_train = 100000                               # number of training sample of NN
N_test = 10                                    # number of test points
load = True                                    # set to False if model has to be retrained
eps = np.finfo(float).eps                      # machine precision
set_param = 'elem'                             # Tube param ('elem' or 'splx')
                                               # Note: simplex param requires delta >= 0.5

# Variables initialisation
N_state = param.x_init.size                    # number of states
N_input = param.u_init.size                    # number of inputs
x = np.zeros((N_state, N+1))                   # state
pos = np.zeros((2, N+1))                       # position
x[:, 0] =  param.x_init
u = np.zeros((N_input, N))                     # control input
u_0 =np.ones((N_input,N))*param.u_init[:,None] # (feasible) guess control input                     
x_0 = np.zeros((N_state, N+1))                 # (feasible) guess trajectory
x_r = np.ones_like(x)*param.h_r[:, None]       # reference state trajectory 
u_r = np.ones_like(u)*param.u_r[:, None]       # reference trajectory
t = np.zeros(N+1)                              # time vector 
K = np.zeros((N, N_input, N_state))            # gain matrix 
Phi1 = np.zeros((N, N_state, N_state))         # closed-loop state transition matrix of f1
Phi2 = np.zeros((N, N_state, N_state))         # closed-loop state transition matrix of f2
real_obj = np.zeros((N, maxIter+1))            # objective value
X_0 = np.zeros((N, maxIter+1, N_state, N+1))   # store state guess trajectories
U_0 = np.zeros((N, maxIter+1, N_input, N))     # store input guess trajectories
X_low = np.zeros((N, maxIter+1, N_state, N+1)) # store perturbed state (lower bound)
X_up = np.zeros((N, maxIter+1, N_state, N+1))  # store perturbed state (upper bound)
S = np.zeros((N, maxIter+1, N_state, N+1))     # store perturbed state    
Q, R = param.Q, param.R                        # penalty matrices

# Activation
if activation == 'relu':
    sigma    = lambda x: np.maximum(x, 0)             # ReLU for numpy
    sigma_cp = lambda x: cp.maximum(x, 0)             # ReLU for cvxpy
    dsigma   = lambda x: np.diag(np.heaviside(x, 0))  # Heaviside
    layer_sigma = ReLU

elif activation == 'elu':
    print('Not implemented yet, abort')
    sys.exit(1)
    sigma  = lambda x: np.where(x>=0, x, np.exp(x)-1)
    dsigma = lambda x: np.diag(np.where(x>=0, 1, np.exp(x)))
    layer_sigma = ELU
else:
    print('Inconsistent activation function, abort')
    sys.exit(1)

# Tube cross section parameterisation
if set_param   == 'elem':
    from optimisation import cvx_opt_elem_fast as cvx_opt    # elementwise bounds
    
elif set_param == 'splx':
    from optimisation import cvx_opt_simplex_fast as cvx_opt # simplex bounds
else:
    print('Inconsistent tube parameterisation, abort')
    sys.exit(1)
    
##########################################################################################
################################### DC decomposition #####################################
##########################################################################################
# Generate training samples
N_state_DC = 1  # number of states in the nonlinear part of the dynamics (1 state: alpha)
N_input_DC = 1  # number of input in the  nonlinear part of the dynamics: (1 input: u1) 
x_train = (param_DC.x_max-param_DC.x_min)*np.random.rand(N_state_DC,N_train)+param_DC.x_min
u_train = (param_DC.u_max-param_DC.u_min)*np.random.rand(N_input_DC,N_train)+param_DC.u_min
y_train = f(x_train, u_train, param)
z_train = np.vstack([x_train, u_train])  # assemble input data
avg, std = 0, 1

# Generate test samples
x_test = (param_DC.x_max-param_DC.x_min)*np.random.rand(N_state_DC, N_test)+param_DC.x_min
u_test = (param_DC.u_max-param_DC.u_min)*np.random.rand(N_input_DC, N_test)+param_DC.u_min
z_test = np.vstack([x_test, u_test])    # assemble input data
y_test = f(x_test, u_test, param)

# DC split
model_f_DC, model_g, model_h = DC.split(N_unit, N_layer, layer_sigma, 
                                        activation, batch_size, epochs, 
                                        z_train, z_test, y_train, y_test, load)

# Define functions of the decomposition from NN weights
weights_g, weights_h = model_g.get_weights(), model_h.get_weights()
g     = lambda x: DC.weight_predict(x, sigma, weights_g)
h     = lambda x: DC.weight_predict(x, sigma, weights_h)

# Plot 
DC.plot(model_f_DC, model_g, model_h, sigma, param_DC)

# Test fit and split
DC.check(f, g, h, z_test, param)

# Sqrt
sqrt_Q = sqrtm(Q)
sqrt_R = sqrtm(R)
Q_lqr =  param.Q_lqr
R_lqr = param.R_lqr

##########################################################################################
############################### Terminal set computation #################################
##########################################################################################

# Compute the terminal set parameters
Q_N, gamma_N, K_hat = term(param, delta, weights_g, weights_h, sigma, dsigma)
sqrt_Q_N = sqrtm(Q_N)
print("Terminal set parameters Q_hat, K_hat, gamma_hat :")
print("Q_N\n", Q_N)
print("K_hat\n", K_hat)
print("gamma_N\n", gamma_N)

##########################################################################################
################################# Feasible trajectory ####################################
##########################################################################################

# Generate a feasible guess trajectory
d_feas = 0.1
x_feas, u_feas, t_feas = feasibility(f_full, x[:, 0], x_r, d_feas, math.floor(T/d_feas), param)
                                                
t_0 = np.arange(N+1)*delta
x_0, u_0 = interp_feas(t_0, t_feas, x_feas, u_feas)

##########################################################################################
###################################### Tests #############################################
##########################################################################################

"""## Test feasibility
#x_feas2 = eul(f_DC, u_feas, x_feas[:, 0], d_feas, param)
    
    
plt.figure()
plt.plot(t_feas, x_feas[0, :], '-b')
plt.plot(t_feas, np.ones_like(t_feas)*param.x_max[0], '--r')
plt.plot(t_feas, np.ones_like(t_feas)*param.x_min[0], '--r')
#plt.plot(t_feas, x_feas2[0, :], '-r')
plt.ylabel('alpha')
plt.xlabel('Time (s)')
        
plt.figure()
plt.plot(t_feas, x_feas[1, :], '-b')
plt.plot(t_feas, np.ones_like(t_feas)*param.x_max[1], '--r')
plt.plot(t_feas, np.ones_like(t_feas)*param.x_min[1], '--r')
#plt.plot(t_feas, x_feas2[1, :], '-r')
plt.ylabel('dy')
plt.xlabel('Time (s)')


plt.figure()
plt.plot(t_feas, x_feas[2, :], '-b')
plt.plot(t_feas, np.ones_like(t_feas)*param.x_max[2], '--r')
plt.plot(t_feas, np.ones_like(t_feas)*param.x_min[2], '--r')
#plt.plot(t_feas, x_feas2[1, :], '-r')
plt.ylabel('dz')
plt.xlabel('Time (s)')

plt.figure()
plt.plot(t_feas, x_feas[3, :], '-b')
plt.plot(t_feas, np.ones_like(t_feas)*param.x_max[3], '--r')
plt.plot(t_feas, np.ones_like(t_feas)*param.x_min[3], '--r')
#plt.plot(t_feas, x_feas2[1, :], '-r')
plt.ylabel('dalpha')
plt.xlabel('Time (s)')

plt.figure()
plt.plot(t_feas[:-1], u_feas[0, :], '-b')
plt.plot(t_feas, np.ones_like(t_feas)*param.u_max[0], '--r')
plt.plot(t_feas, np.ones_like(t_feas)*param.u_min[0], '--r')
#plt.plot(t_feas, x_feas2[1, :], '-r')
plt.ylabel('u1')
plt.xlabel('Time (s)')

plt.figure()
plt.plot(t_feas[:-1], u_feas[1, :], '-b')
plt.plot(t_feas, np.ones_like(t_feas)*param.u_max[1], '--r')
plt.plot(t_feas, np.ones_like(t_feas)*param.u_min[1], '--r')
#plt.plot(t_feas, x_feas2[1, :], '-r')
plt.ylabel('u2')
plt.xlabel('Time (s)')

plt.show()"""

## Test linearisation
"""A1, B1, A2, B2 = linearise(x_0[:, :-1], u_0, weights_g, weights_h, sigma, dsigma) 
A = A1 - A2
B = B1 - B2

A_true, B_true = linearise_true(x_0[:, :-1], u_0, param)

for l in range(N-1):
    print("New batch: ")
    print("A_true", A_true[l,:,:])
    print("A", A[l,:,:])
    print("B_true", B_true[l,:,:])
    print("B", B[l,:,:])
    print("dA: ", np.linalg.norm(A_true[l,:,:] - A[l,:,:]))
    print("dB :", np.linalg.norm(B_true[l,:,:] - B[l,:,:]))"""
    
##########################################################################################
####################################### TMPC loop ########################################
##########################################################################################
avg_iter_time = 0
iter_count = 0

for i in range(N):

    print("Computation at time step {}/{}...".format(i+1, N)) 
    
    # Guess trajectory update
    if i > 0:
        #x_0[:, :-1] = x_0[:, 1:]
        x_0[:, :-1] = eul(f_full, u_0[:, :-1], x[:, i], delta, param) 
        A1_hat, B1_hat, A2_hat, B2_hat = linearise(x_0[:, -2, None], param.u_r[:, None], 
                                                   weights_g, weights_h, sigma, dsigma)  
        
        A_hat, B_hat = discretise(A1_hat - A2_hat, B1_hat - B2_hat, delta) # discrete-time
        
        K_hat, _ = dp(A_hat[0, :, :], B_hat[0, :, :], Q, R, Q_N)
        u_0[:, -1, None] = K_hat @ ( x_0[:,-2, None]  - x_r[:, -2, None])\
                                                      + param.u_r[:, None]        # term u
        x_0[:, -1]  = x_0[:, -2] + delta*(f_full(x_0[:, -2], u_0[:, -1] , param)) # term x 
    else:
        pass

    # Iteration
    k = 0 
    real_obj[i, 0] = 5000 
    delta_obj = 5000
    print('{0: <6}'.format('iter'), '{0: <5}'.format('status'), 
          '{0: <18}'.format('time'), '{}'.format('cost'))
    while k < maxIter:
    #while real_obj[i, k] > tol1 and k < maxIter and delta_obj > 0.1:
        
        # Linearise system at x_0, u_0
        A1, B1, A2, B2 = linearise(x_0[:, :-1], u_0, weights_g, weights_h, sigma, dsigma) 
                                                                      # continuous-time DC  
        
        A, B = discretise(A1 - A2, B1 - B2, delta)                    # discrete-time
    
        #print(x_0)
        
        # Compute K matrix (using dynamic programming)
        P = Q_N
        for l in reversed(range(N)): 
            K[l, :, :], P = dp(A[l, :, :], B[l, :, :], Q_lqr, R_lqr, P)
        
        # Prediction over trajectory
        z_0 = np.vstack([x_0[0, :-1], u_0[0, :]])  # stack data
        g_0 = g(z_0)
        h_0 = h(z_0)
        g_cvx = lambda x: DC.weight_predict(x, sigma_cp, weights_g)
        h_cvx = lambda x: DC.weight_predict(x, sigma_cp, weights_h)
        
        ##################################################################################
        ############################ Optimisation problem ################################
        ##################################################################################
        
        t_start = time.time()
        problem, X_lb, X_ub, v = cvx_opt(x[:, i], x_0, u_0, x_r, u_r, delta, param, 
                                         sqrt_Q, sqrt_R, sqrt_Q_N, gamma_N, K, A1,
                                         A2, B1, B2, 0, 0, g_cvx, h_cvx, g_0, h_0)

        iter_time = time.time()-t_start
        avg_iter_time += iter_time
        
        print('{0: <5}'.format(k+1), '{0: <5}'.format(problem.status), 
              '{0: <5.2f}'.format(iter_time), '{0: <5}'.format(problem.value))
        if problem.status not in ["optimal"] and k > 0:
            print("Problem status {} at iteration k={}".format(problem.status, k))
            break
        
        ##################################################################################
        ############################### Iteration update #################################
        ##################################################################################
        # Save variables 
        X_low[i, k, :, :] = X_lb.value.copy()
        X_up[i, k, :, :] = X_ub.value.copy()
        X_0[i, k, :, :] = x_0.copy()
        U_0[i, k, :, :] = u_0.copy()
        x_0_old = x_0.copy()
        f_x = x_0.copy()
 
        # Input and state update
        s = np.zeros((N_state, N+1))   # state perturbation s = x - x_0
        s[:, 0] = x[:, i] - x_0[:, 0]  # implies s_0 = 0
        Kx = np.zeros_like(v.value)
        for l in range(N):
            Kx[:, l] =   K[l, :, :] @ x_0[:, l]
            u_0[:, l] = v.value[:, l] + Kx[:, l]
            f_x[:, l+1] = eul(f_full, u_0[:, l], x_0[:, l], delta, param)
            x_0[:, l+1] =f_x[:, l+1]
            s[:, l+1] = x_0[:, l+1]-x_0_old[:, l+1]
           
        S[i, k, :, :] = s.copy()
        
        # Step update 
        k += 1
        iter_count += 1
        real_obj[i, k] = problem.value
        delta_obj = real_obj[i, k-1]-real_obj[i, k]
        
    ######################################################################################
    #################################### System update ###################################
    ######################################################################################
    # Uncomment to exit at first iteration 
    """x = x_0
    u = u_0
    t = np.cumsum(np.ones(x.shape[1])*delta)-delta
    pos = np.hstack([np.zeros((2, 1)), np.cumsum(delta*x[1:3, :-1], axis=1)])
    x_r_0 = x_r
    break"""
    
    u[:, i] = u_0[:, 0]                                      # apply first input
    u_0[:, :-1] = u_0[:, 1:]                                 # extract tail of the input
    x[:, i+1] = eul(f_full, u[:, i], x[:, i], delta, param)  # update nonlinear dynamics 
    t[i+1] = t[i] + delta
    pos[:, i+1] = pos[:, i] + delta*x[1:3, i]
    print('State (a, dy, dz, da):', x[:, i], 'Input (u1, u2):', u[:, i])


##########################################################################################
##################################### Plot results #######################################
##########################################################################################
print('Average time per iteration: ', avg_iter_time/iter_count)
#print('Average time per time step: ', avg_iter_time/i)

if not os.path.isdir('plot'):
    os.mkdir('plot')
     
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['axes.unicode_minus'] = False
def math_formatter(x, pos):
    return "${}$".format(x).replace("-", u"\u2212")

## Comparison solution with initial guess
fig, axs = plt.subplots(3, 2)
axs[0, 0].plot(t, x[0,:]*180/np.pi, label=r'$\alpha$')
axs[0, 0].plot(t, X_0[0, 0, 0, :]*180/np.pi,label=r'$\alpha^0$')
axs[0, 0].plot(t, x_r[0,:]*180/np.pi, '-.', label=r'$\alpha^r$')
axs[0, 0].legend(loc='lower right', prop={'size': 10})
axs[0, 0].set_title(r'$\alpha$ (deg)')
axs[0, 0].set_xticks([])

axs[1, 0].plot(t, x[1,:], label=r'$\dot{y}$')
axs[1, 0].plot(t, X_0[0, 0, 1, :], label=r'$\dot{y}^0$')
axs[1, 0].plot(t, x_r[1,:], '-.', label=r'$\dot{y}^r$')
axs[1, 0].legend(loc='lower right', prop={'size': 10})
axs[1, 0].set_title('Horizontal velocity (m/s)')
axs[1, 0].set_xticks([])

axs[2, 0].plot(t[:-1], u[0,:], label=r'$u_1$')
axs[2, 0].plot(t[:-1], U_0[0, 0, 0, :], label=r'$u_1^0$')
axs[2, 0].legend(loc='upper right', prop={'size': 10})
axs[2, 0].set_title(r'$u_1$')
axs[2, 0].set(xlabel='Time (s)')


axs[0, 1].plot(t[:-1], u[1,:], label=r'$u_2$')
axs[0, 1].plot(t[:-1], U_0[0, 0, 1, :], label=r'$u_2^0$')
axs[0, 1].legend(loc='upper right', prop={'size': 10})
axs[0, 1].set_title(r'$u_2$')
axs[0, 1].set_xticks([])

axs[1, 1].plot(t, x[2,:], label=r'$\dot{z}$')
axs[1, 1].plot(t, X_0[0, 0, 2, :], label=r'$\dot{z}^0$')
axs[1, 1].plot(t, x_r[2,:], '-.', label=r'$\dot{z}^r$')
axs[1, 1].legend(loc='lower right', prop={'size': 10})
axs[1, 1].set_title('Vertical velocity (m/s)')
axs[1, 1].set_xticks([])

axs[2, 1].plot(t, -pos[1, :], label=r'$h$')
axs[2, 1].set_title('Relative altitude (m)')
axs[2, 1].legend(loc='lower right', prop={'size': 10})
axs[2, 1].set(xlabel='Time (s)')

fig.tight_layout()
plt.savefig('plot/tmpc1.eps', format='eps')
plt.savefig('plot/tmpc1.pdf', format='pdf')


## Objective value
obj_init = seed_cost(X_0[0, 0, :, :], U_0[0, 0, :, :], Q, R, Q_N, param)
plt.figure()
plt.semilogy(range(0, N+1), np.hstack([obj_init, real_obj[:, 1]]))
plt.ylabel('Objective value $J$ at first iteration (-)')
plt.xlabel('Time step n (-)') 
plt.savefig('plot/obj1.eps', format='eps')
plt.savefig('plot/obj1.pdf', format='pdf')

## Objective value at first iteration
obj_init = seed_cost(X_0[0, 0, :, :], U_0[0, 0, :, :], Q, R, Q_N, param)
plt.figure()
plt.semilogy(range(0, maxIter+1), np.hstack([obj_init, real_obj[0, 1:]]))
plt.ylabel('Objective value $J$ at first time step (-)')
plt.xlabel('Iteration k (-)')

## Save data
steps = range(0, N+1)
objs = np.hstack([obj_init, real_obj[:, 1]])
iter = range(0, maxIter+1)
objs0 = np.hstack([obj_init, real_obj[0, 1:]])
np.savez('data_NN.npz',iter= iter, objs0=objs0, steps=steps, objs=objs, t=t, x=x, 
                                                  X_0=X_0, x_r=x_r, u=u, U_0=U_0, pos=pos)
plt.show()