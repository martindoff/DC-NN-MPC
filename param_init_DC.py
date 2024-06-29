""" Parameters of the PVTOL nonlinear dynamics (subset of the full dynamics)

x2' = (g + u1)*sin(x1)             ]             f1 = (g + u)*sin(x)
                                   ] ==> 
x3' = (g + u1)*cos(x1) - g         ]             f2 = (g + u)*cos(x) - g

(c) Martin Doff-Sotta, University of Oxford (martin.doff-sotta@eng.ox.ac.uk)

"""
import numpy as np
import param_init as param

## Physical parameters
g = param.g                                         # Gravity acceleration (m/s^2) 

## State and input constraints
u_max = param.u_max[0]                              # max input
u_min = -u_max                                      # min input
x_max = param.x_max[0]                              # max state
x_min = -x_max                                      # min state
ctr = np.array([[x_min, x_max], 
                [u_min, u_max]])                    # gather all in one array
                

## Nonlinear dynamics
f1 = lambda alpha, u: (u + g)*np.sin(alpha)
f2 = lambda alpha, u: (u + g)*np.cos(alpha) - g