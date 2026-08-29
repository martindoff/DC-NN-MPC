""" Solve the DC-TMPC problem for several curvature penalty weights and plot the results

For each weight the closed loop is simulated by calling main.py, and two figures are
produced at the end:

- the distribution of the linearisation error of g and h over steps of the size of the
  tube cross section, which is the quantity the curvature penalty acts on;
- the region of attraction of the closed loop, that is the set of initial states from
  which the MPC problem is feasible, obtained by bisection along rays from the origin.
  A Monte Carlo estimate of the feasible fraction of the state box is computed as well
  if --mc is given.

Run:  python3 run_sweep.py
      python3 run_sweep.py --lambdas 0 1.5 --penalise h --mc 500

(c) 08/2026 - Martin Doff-Sotta, University of Oxford (martin.doff-sotta@eng.ox.ac.uk)

"""
import argparse
import math
import os
import subprocess
import sys

import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm

import matplotlib
import matplotlib.pyplot as plt

try:
    from keras.src.layers import ReLU
except ImportError:
    from keras.layers import ReLU

import DC_decomposition as DC
import param_init as param
import param_init_DC as param_DC
from control_custom import dp
from pvtol_model import f, linearise, discretise, feasibility, f_full, interp_feas
from terminal import get_term
from optimisation import cvx_opt_elem_fast as cvx_opt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

parser = argparse.ArgumentParser(description='Solve the DC-TMPC problem for several '
                                             'curvature penalty weights')
parser.add_argument('--lambdas', type=float, nargs='+', default=[0., 1.5],
                    help='curvature penalty weights (default 0 and 1.5)')
parser.add_argument('--penalise', choices=DC.PENALISE_MODES, default='h',
                    help='networks whose curvature was penalised (default h)')
parser.add_argument('--rays', type=int, default=16,
                    help='number of directions per slice of the region of attraction')
parser.add_argument('--bisect', type=int, default=7,
                    help='bisection steps along each direction')
parser.add_argument('--mc', type=int, default=0,
                    help='Monte Carlo samples for the feasible fraction of the state box '
                         '(0: not computed, one MPC problem is solved per sample)')
parser.add_argument('--skip-runs', action='store_true',
                    help='only produce the figures, from the models already trained')
args = parser.parse_args()

##########################################################################################
################################### Closed-loop runs #####################################
##########################################################################################
if not args.skip_runs:
    for lam in args.lambdas:
        if not os.path.isfile(DC.weights_file(lam)):
            print('missing decomposition {} - run train_sweep.py first'.format(
                  DC.weights_file(lam)))
            sys.exit(1)

        cmd = [sys.executable, 'main.py', '--lam', repr(lam),
               '--penalise', args.penalise, '--no-plots',
               '--outfile', 'data_lam{:g}.npz'.format(lam)]
        print('\n*************** {} ***************'.format(' '.join(cmd)))
        if subprocess.run(cmd).returncode != 0:
            print('run failed for lam = {:g}'.format(lam))
            sys.exit(1)

##########################################################################################
################################ Problem set-up for the figures ##########################
##########################################################################################
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


models = {lam: load_decomposition(lam) for lam in args.lambdas}
colors = plt.cm.viridis(np.linspace(0, 0.8, len(args.lambdas)))

##########################################################################################
############################# Distribution of the linearisation error ####################
##########################################################################################
n_samp = 20000
rng = np.random.default_rng(0)
x_s = (param_DC.x_max - param_DC.x_min)*rng.random((1, n_samp)) + param_DC.x_min
u_s = (param_DC.u_max - param_DC.u_min)*rng.random((1, n_samp)) + param_DC.u_min
z_s = np.vstack([x_s, u_s])
d_s = (2*rng.random((2, n_samp)) - 1)*np.array(DC.HESS_STEP)[:, None]

tex = [r'$\ddot{y}$', r'$\ddot{z}$']
fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharex='col')
for row, name in enumerate(['h', 'g']):
    for col in range(2):
        ax = axs[row, col]
        for i, lam in enumerate(args.lambdas):
            w = models[lam][1] if name == 'h' else models[lam][0]
            e = DC.lin_err(w, z_s, d_s)[col]
            ax.hist(e, bins=60, histtype='step', density=True, color=colors[i],
                    label=r'$\lambda$ = {:g} (mean {:.3f})'.format(lam, e.mean()))
        ax.set_yscale('log')
        ax.set_title(r'$e_{}$ ({})'.format(name, tex[col]), fontsize=10)
        ax.legend(prop={'size': 8})
        if row == 1:
            ax.set_xlabel(r'$e$ (m/s$^2$)')
        if col == 0:
            ax.set_ylabel('density')
fig.suptitle('Linearisation error over steps of the size of the tube cross section')
fig.tight_layout()
fig.savefig('plot/lin_err.pdf')
fig.savefig('plot/lin_err.png', dpi=200)
print('\nLinearisation error written to plot/lin_err.pdf')

##########################################################################################
################################## Region of attraction ##################################
##########################################################################################
SLICES = {'a-da': ((0, 3), (r'$\alpha_0$ (rad)', r'$\dot{\alpha}_0$ (rad/s)')),
          'a-dy': ((0, 1), (r'$\alpha_0$ (rad)', r'$\dot{y}_0$ (m/s)'))}


def is_feasible(x_init, weights, term):
    """ True if the MPC problem is feasible at the first time step from x_init """
    w_g, w_h = weights
    Q_N, gamma_N, sqrt_Q_N = term
    try:
        x_feas, u_feas, _ = feasibility(f_full, x_init, x_r, d_feas,
                                        math.floor(T/d_feas), param)
        x_0, u_0 = interp_feas(t_0, t_feas, x_feas, u_feas)

        A1, B1, A2, B2 = linearise(x_0[:, :-1], u_0, w_g, w_h, sigma, dsigma)
        A, B = discretise(A1 - A2, B1 - B2, delta)

        K = np.zeros((N, N_input, N_state))
        P = Q_N
        for l in reversed(range(N)):
            K[l, :, :], P = dp(A[l, :, :], B[l, :, :], param.Q_lqr, param.R_lqr, P)

        z_0 = np.vstack([x_0[0, :-1], u_0[0, :]])
        g_0 = DC.weight_predict(z_0, sigma, w_g)
        h_0 = DC.weight_predict(z_0, sigma, w_h)
        g_cvx = lambda v: DC.weight_predict(v, sigma_cp, w_g)
        h_cvx = lambda v: DC.weight_predict(v, sigma_cp, w_h)

        problem, _, _, _ = cvx_opt(x_init, x_0, u_0, x_r, u_r, delta, param, sqrt_Q,
                                   sqrt_R, sqrt_Q_N, gamma_N, K, A1, A2, B1, B2,
                                   param.W_low, param.W_up, g_cvx, h_cvx, g_0, h_0)
        return problem.status == 'optimal'
    except Exception:
        return False


def boundary(weights, term, key):
    """ Boundary of a slice of the region of attraction, by bisection along rays """
    idx, _ = SLICES[key]
    th = np.linspace(0, np.pi, args.rays, endpoint=False)
    radii = np.zeros_like(th)
    for j, angle in enumerate(th):
        direction = np.array([np.cos(angle), np.sin(angle)])
        lo, hi = 0., np.inf
        for k, i in enumerate(idx):     # largest radius inside the state box
            if abs(direction[k]) > 1e-12:
                lim = param.x_max[i] if direction[k] > 0 else -param.x_min[i]
                hi = min(hi, lim/abs(direction[k]))
        x_init = np.zeros(N_state)
        x_init[list(idx)] = hi*direction
        if is_feasible(x_init, weights, term):
            radii[j] = hi
            continue
        for _ in range(args.bisect):
            mid = 0.5*(lo + hi)
            x_init[list(idx)] = mid*direction
            if is_feasible(x_init, weights, term):
                lo = mid
            else:
                hi = mid
        radii[j] = lo

    # the dynamics are unchanged by (alpha, dy, dalpha) -> -(alpha, dy, dalpha), so the
    # boundary is symmetric about the origin and only half of the directions are needed
    return np.concatenate([th, th + np.pi]), np.concatenate([radii, radii])


def feasible_fraction(weights, term, n):
    """ Fraction of the state box from which the problem is feasible, with its 95%
    confidence interval """
    rg = np.random.default_rng(0)
    X = param.x_min[:, None] + (param.x_max - param.x_min)[:, None]*rg.random((N_state, n))
    ok = np.array([is_feasible(X[:, i], weights, term) for i in range(n)])
    p = ok.mean()

    return p, 1.96*np.sqrt(max(p*(1 - p), 1e-12)/n)


results, fractions = {}, {}
for lam in args.lambdas:
    w_g, w_h = models[lam]
    Q_N, gamma_N, _ = get_term(param, delta, w_g, w_h, sigma, dsigma)
    term = (Q_N, gamma_N, sqrtm(Q_N))
    print('\n*************** region of attraction, lam = {:g} ***************'.format(lam))
    print('gamma_N = {:.4f}'.format(gamma_N))
    results[lam] = {key: boundary((w_g, w_h), term, key) for key in SLICES}
    if args.mc > 0:
        fractions[lam] = feasible_fraction((w_g, w_h), term, args.mc)
        print('feasible fraction of the state box = {:.3f} +/- {:.3f}'.format(
              *fractions[lam]))

fig, axs = plt.subplots(1, len(SLICES), figsize=(4.6*len(SLICES), 4.2), squeeze=False)
for col, key in enumerate(SLICES):
    ax = axs[0, col]
    idx, (xl, yl) = SLICES[key]
    for i, lam in enumerate(args.lambdas):
        th, r = results[lam][key]
        o = np.argsort(th)
        x, y = r[o]*np.cos(th[o]), r[o]*np.sin(th[o])
        ax.plot(np.append(x, x[0]), np.append(y, y[0]), '-o', ms=3, color=colors[i],
                label=r'$\lambda$ = {:g}'.format(lam))
    for v in (param.x_max[idx[0]], param.x_min[idx[0]]):
        ax.axvline(v, color='r', ls='--', lw=0.8)
    for v in (param.x_max[idx[1]], param.x_min[idx[1]]):
        ax.axhline(v, color='r', ls='--', lw=0.8)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', prop={'size': 9})

title = 'Region of attraction (feasible initial states at the first time step)'
if fractions:
    title += '\n' + ', '.join(r'$\lambda$={:g}: {:.3f}$\pm${:.3f}'.format(l, *fractions[l])
                              for l in args.lambdas)
fig.suptitle(title, fontsize=10)
fig.tight_layout()
fig.savefig('plot/feasible_region.pdf')
fig.savefig('plot/feasible_region.png', dpi=200)
print('\nRegion of attraction written to plot/feasible_region.pdf')
plt.show()
