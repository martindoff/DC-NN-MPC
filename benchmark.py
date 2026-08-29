""" Compare the closed loop obtained with two curvature penalty weights

The two decompositions are compared from the same set of initial conditions, so that the
difference between them is not masked by the spread between initial conditions. For each
initial condition the closed loop is simulated and the following quantities are recorded:

- the number of iterations of the convex-concave procedure per time step, the iteration
  being stopped once the objective decrement falls below a tolerance;
- the width of the tube cross section and the slack between the tube and the state
  constraints, both measures of the conservatism of the scheme;
- the computation time per iteration and the closed-loop cost.

Run:  python3 benchmark.py --lambdas 0 1.5 --n-ic 25 --n-steps 15

(c) 08/2026 - Martin Doff-Sotta, University of Oxford (martin.doff-sotta@eng.ox.ac.uk)

"""
import argparse
import math
import sys
import time

import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm

try:
    from keras.src.layers import ReLU
except ImportError:
    from keras.layers import ReLU

import DC_decomposition as DC
import param_init as param
from control_custom import eul, dp
from pvtol_model import linearise, discretise, feasibility, f_full, interp_feas
from terminal import get_term
from optimisation import cvx_opt_elem_fast as cvx_opt

parser = argparse.ArgumentParser(description='Compare the closed loop for two curvature '
                                             'penalty weights')
parser.add_argument('--lambdas', type=float, nargs='+', default=[0., 1.5],
                    help='curvature penalty weights to compare (default 0 and 1.5)')
parser.add_argument('--penalise', choices=DC.PENALISE_MODES, default='h',
                    help='networks whose curvature was penalised (default h)')
parser.add_argument('--n-ic', type=int, default=25,
                    help='number of initial conditions, common to both weights')
parser.add_argument('--n-steps', type=int, default=15,
                    help='closed-loop time steps per run')
parser.add_argument('--maxiter', type=int, default=10,
                    help='maximum number of iterations per time step')
parser.add_argument('--tol', type=float, default=0.05,
                    help='the iteration stops once the objective decrement is below this')
parser.add_argument('--ic-scale', type=float, default=0.5,
                    help='initial conditions are drawn in this fraction of the state box')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

## Problem set-up (as in main.py)
N = 30                                         # horizon
T = 15                                         # terminal time
delta = T/N                                    # time step
d_feas = 0.1                                   # step of the feasible trajectory
N_state, N_input = param.x_init.size, param.u_init.size
Q, R = param.Q, param.R
sqrt_Q, sqrt_R = sqrtm(Q), sqrtm(R)
x_r = np.ones((N_state, N+1))*param.h_r[:, None]
u_r = np.ones((N_input, N))*param.u_r[:, None]
t_0 = np.arange(N+1)*delta
t_feas = np.arange(math.floor(T/d_feas)+1)*d_feas

sigma = lambda x: np.maximum(x, 0)
sigma_cp = lambda x: cp.maximum(x, 0)
dsigma = lambda x: np.diag(np.heaviside(x, 0))

N_unit, N_layer = 8, 1
z_void = np.zeros((2, 3))


def load_decomposition(lam):
    """ Weights of the decomposition trained with penalty weight lam """
    _, model_g, model_h = DC.split(N_unit, N_layer, ReLU, 'relu', 32, 1, z_void, z_void,
                                   z_void, z_void, True, lam=lam,
                                   penalise=args.penalise,
                                   file_name=DC.weights_file(lam))
    return model_g.get_weights(), model_h.get_weights()


def run_closed_loop(x_init, weights, term):
    """ Simulate the closed loop from x_init and return the recorded quantities """
    w_g, w_h = weights
    Q_N, gamma_N, sqrt_Q_N = term

    x = np.zeros((N_state, N+1))
    x[:, 0] = x_init
    u = np.zeros((N_input, N))

    x_feas, u_feas, _ = feasibility(f_full, x_init, x_r, d_feas,
                                    math.floor(T/d_feas), param)
    x_0, u_0 = interp_feas(t_0, t_feas, x_feas, u_feas)

    g_cvx = lambda v: DC.weight_predict(v, sigma_cp, w_g)
    h_cvx = lambda v: DC.weight_predict(v, sigma_cp, w_h)
    iters, widths, margins, times = [], [], [], []

    for i in range(args.n_steps):
        if i > 0:                                   # update of the guess trajectory
            x_0[:, :-1] = eul(f_full, u_0[:, :-1], x[:, i], delta, param)
            A1h, B1h, A2h, B2h = linearise(x_0[:, -2, None], param.u_r[:, None],
                                           w_g, w_h, sigma, dsigma)
            A_h, B_h = discretise(A1h - A2h, B1h - B2h, delta)
            K_h, _ = dp(A_h[0], B_h[0], Q, R, Q_N)
            u_0[:, -1, None] = K_h @ (x_0[:, -2, None] - x_r[:, -2, None]) \
                               + param.u_r[:, None]
            x_0[:, -1] = x_0[:, -2] + delta*f_full(x_0[:, -2], u_0[:, -1], param)

        k, prev, n_it = 0, np.inf, 0
        while k < args.maxiter:
            A1, B1, A2, B2 = linearise(x_0[:, :-1], u_0, w_g, w_h, sigma, dsigma)
            A, B = discretise(A1 - A2, B1 - B2, delta)

            K = np.zeros((N, N_input, N_state))
            P = Q_N
            for l in reversed(range(N)):
                K[l, :, :], P = dp(A[l, :, :], B[l, :, :], param.Q_lqr, param.R_lqr, P)

            z_0 = np.vstack([x_0[0, :-1], u_0[0, :]])
            g_0 = DC.weight_predict(z_0, sigma, w_g)
            h_0 = DC.weight_predict(z_0, sigma, w_h)

            t_start = time.time()
            try:
                problem, X_lb, X_ub, v = cvx_opt(x[:, i], x_0, u_0, x_r, u_r, delta,
                                                 param, sqrt_Q, sqrt_R, sqrt_Q_N,
                                                 gamma_N, K, A1, A2, B1, B2,
                                                 param.W_low, param.W_up,
                                                 g_cvx, h_cvx, g_0, h_0)
            except Exception:
                return None
            times.append(time.time() - t_start)

            if problem.status not in ['optimal'] or X_lb.value is None:
                if n_it == 0:
                    return None                     # no solution at the first iteration
                break

            n_it += 1
            W = X_ub.value - X_lb.value
            widths.append(W[1:3, :].sum())
            margins.append(min((param.x_max[:, None] - X_ub.value[:, :-1]).min(),
                               (X_lb.value[:, :-1] - param.x_min[:, None]).min()))

            for l in range(N):                      # input and state update
                u_0[:, l] = v.value[:, l] + K[l] @ x_0[:, l]
                x_0[:, l+1] = eul(f_full, u_0[:, l], x_0[:, l], delta, param)

            dec = prev - problem.value
            prev = problem.value
            k += 1
            if dec < args.tol:
                break

        iters.append(n_it)
        u[:, i] = u_0[:, 0]
        u_0[:, :-1] = u_0[:, 1:]
        x[:, i+1] = eul(f_full, u[:, i], x[:, i], delta, param)

    J = float(sum((x[:, i] - param.h_r) @ Q @ (x[:, i] - param.h_r)
                  + (u[:, i] - param.u_r) @ R @ (u[:, i] - param.u_r)
                  for i in range(args.n_steps)))

    return dict(iters=np.mean(iters), width=np.mean(widths), margin=np.mean(margins),
                time=np.mean(times), J_cl=J)


## Initial conditions, the same for both weights
rng = np.random.default_rng(args.seed)
ICs = [param.x_init.copy()]
lo, hi = args.ic_scale*param.x_min, args.ic_scale*param.x_max
for _ in range(args.n_ic - 1):
    ICs.append(lo + (hi - lo)*rng.random(N_state))

metrics = ['iters', 'width', 'margin', 'time', 'J_cl']
res = {lam: {m: [] for m in metrics} for lam in args.lambdas}
usable = np.ones(len(ICs), dtype=bool)

for lam in args.lambdas:
    print('\n*************** lam = {:g} ***************'.format(lam))
    weights = load_decomposition(lam)
    Q_N, gamma_N, _ = get_term(param, delta, weights[0], weights[1], sigma, dsigma)
    term = (Q_N, gamma_N, sqrtm(Q_N))
    print('gamma_N = {:.4f}'.format(gamma_N))

    for j, x_init in enumerate(ICs):
        out = run_closed_loop(x_init, weights, term)
        if out is None:
            usable[j] = False
            for m in metrics:
                res[lam][m].append(np.nan)
            print('  initial condition {:2d}: infeasible'.format(j))
            continue
        for m in metrics:
            res[lam][m].append(out[m])
        print('  initial condition {:2d}: iterations/step {:.2f} | tube {:.4f} '
              '| cost {:8.2f} | {:.2f} s/iteration'.format(j, out['iters'], out['width'],
                                                           out['J_cl'], out['time']))

## Comparison over the initial conditions feasible for both weights
print('\n' + '='*78)
print('Comparison over {} initial conditions ({} feasible for both weights)'.format(
      len(ICs), int(usable.sum())))
print('='*78)
print(('{:<10}' + '{:>13}'*len(metrics)).format('lam', *metrics))
for lam in args.lambdas:
    values = [np.nanmean(np.array(res[lam][m])[usable]) for m in metrics]
    print(('{:<10g}' + '{:>13.4f}'*len(metrics)).format(lam, *values))

ref = args.lambdas[0]
for lam in args.lambdas[1:]:
    print('\nChange with respect to lam = {:g}:'.format(ref))
    for m in metrics:
        a = np.array(res[ref][m])[usable]
        b = np.array(res[lam][m])[usable]
        print('  {:<8} {:+9.4f} ({:+6.1f} %)'.format(m, (b - a).mean(),
                                                     100*(b - a).mean()/abs(a.mean())))
