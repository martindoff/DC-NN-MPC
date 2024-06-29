""" Model parameters of the PVTOL aircraft

Let (x1, x2, x3, x4) = (alpha, y', z', alpha'), the state space model is: 

x1' = x4
x2' = (g + u1)*sin(x1)
x3' = (g + u1)*cos(x1) - g
x4' = u2

(c) 05/2024 - Martin Doff-Sotta, University of Oxford (martin.doff-sotta@eng.ox.ac.uk)

"""
from pvtol_model import f_full
import numpy as np
import math

## Vehicle parameters
g = 9.81                                  # Gravity acceleration (m/s^2)

## State and input constraints
u_max = np.array([10, 10])                # max input
u_min = -u_max                            # min input
x_max = np.array([.3, 30, 10, 1])         # max state
x_min = -x_max                            # min state
x_init = np.array([0.1, 0, 0, 0])
u_init = np.array([0, 0])

## State and input reference trajectory
h_r = np.array([0, 0, 0, 0])
u_r = np.array([0, 0])

## Wind gust parameters
#W_low, W_up = 0, 0
W_up = np.array([30, 10])/100
W_low = -W_up

## State and input penalty matrices
Q = np.diag([10, 1, 1, 1])                # State penalty matrix
R = np.diag([.1, 1])/1000                 # Input penalty matrix
 
## LQR penalty matrices
Q_lqr = Q
R_lqr = R

## Terminal set parameters
u_term = np.array([1, 1])                 # Terminal set bound on input 
x_term = np.array([0.03, 1, 1, 0.1])      # Terminal set bound on state 