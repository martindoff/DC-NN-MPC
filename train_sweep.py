""" Train the DC decomposition for several curvature penalty weights

The curvature of the decomposition is penalised at training stage as described in
DC_decomposition.hessian_penalty. One decomposition is trained per weight lam, with the
same initialisation and the same training data, so that the weights can be compared. A
weight of 0 gives the unpenalised decomposition.

Run:  python3 train_sweep.py
      python3 train_sweep.py --lambdas 0 1.5 --penalise h

(c) 08/2026 - Martin Doff-Sotta, University of Oxford (martin.doff-sotta@eng.ox.ac.uk)

"""
import argparse
import csv

import numpy as np
import keras

try:
    from keras.src.layers import ReLU
except ImportError:
    from keras.layers import ReLU

import DC_decomposition as DC
import param_init as param
import param_init_DC as param_DC
from pvtol_model import f

# Training parameters (as in main.py)
SEED = 0
N_unit = 8                                     # number of units of neural network (NN)
N_layer = 1                                    # number of hidden layers of NN
batch_size = 32                                # NN training batch size
epochs = 200                                   # NN training epochs
N_train = 100000                               # number of training sample of NN
N_test = 2000                                  # number of test points

parser = argparse.ArgumentParser(description='Train the DC decomposition for several '
                                             'curvature penalty weights')
parser.add_argument('--lambdas', type=float, nargs='+', default=[0., 1.5],
                    help='curvature penalty weights (default 0 and 1.5)')
parser.add_argument('--penalise', choices=DC.PENALISE_MODES, default='h',
                    help='networks whose curvature is penalised (default h, the function '
                         'linearised in the tube constraint)')
parser.add_argument('--epochs', type=int, default=epochs)
parser.add_argument('--csv', type=str, default='training_metrics.csv',
                    help='file in which the metrics of each weight are written')
args = parser.parse_args()

sigma = lambda x: np.maximum(x, 0)

## Training and test samples, common to every penalty weight
rng = np.random.default_rng(SEED)
x_train = (param_DC.x_max - param_DC.x_min)*rng.random((1, N_train)) + param_DC.x_min
u_train = (param_DC.u_max - param_DC.u_min)*rng.random((1, N_train)) + param_DC.u_min
y_train = f(x_train, u_train, param)
z_train = np.vstack([x_train, u_train])

x_test = (param_DC.x_max - param_DC.x_min)*rng.random((1, N_test)) + param_DC.x_min
u_test = (param_DC.u_max - param_DC.u_min)*rng.random((1, N_test)) + param_DC.u_min
y_test = f(x_test, u_test, param)
z_test = np.vstack([x_test, u_test])

## Points at which the curvature of the trained models is measured
x_c = (param_DC.x_max - param_DC.x_min)*rng.random((1, N_test)) + param_DC.x_min
u_c = (param_DC.u_max - param_DC.u_min)*rng.random((1, N_test)) + param_DC.u_min
z_c = np.vstack([x_c, u_c])


def curvature(weights, step=DC.HESS_STEP):
    """ Mean curvature sum_i d_i^2 [H]_ii of a network over the input box, per output """
    f0 = DC.weight_predict(z_c, sigma, weights)
    tot = np.zeros_like(f0)
    for i in range(z_c.shape[0]):
        e = np.zeros_like(z_c)
        e[i, :] = step[i]
        tot += (DC.weight_predict(z_c + e, sigma, weights) - 2*f0
                + DC.weight_predict(z_c - e, sigma, weights))

    return tot.mean(axis=1)


rows = []
for lam in args.lambdas:
    file_name = DC.weights_file(lam)
    print('\n*************** lam = {:g} -> {} ***************'.format(lam, file_name))
    keras.utils.set_random_seed(SEED)   # same initialisation for every weight
    _, model_g, model_h = DC.split(N_unit, N_layer, ReLU, 'relu', batch_size,
                                   args.epochs, z_train, z_test, y_train, y_test, False,
                                   lam=lam, penalise=args.penalise, file_name=file_name)

    w_g, w_h = model_g.get_weights(), model_h.get_weights()
    pred = DC.weight_predict(z_test, sigma, w_g) - DC.weight_predict(z_test, sigma, w_h)
    mae = np.abs(pred - y_test).mean()
    c_g, c_h = curvature(w_g), curvature(w_h)

    rows.append(dict(lam=lam, mae=mae,
                     curv_g_ddy=c_g[0], curv_g_ddz=c_g[1],
                     curv_h_ddy=c_h[0], curv_h_ddz=c_h[1]))
    print('lam = {:g}: MAE = {:.4f} | curvature h = [{:.3f}, {:.3f}] '
          '| curvature g = [{:.3f}, {:.3f}]'.format(lam, mae, c_h[0], c_h[1],
                                                    c_g[0], c_g[1]))

with open(args.csv, 'w', newline='') as fp:
    writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print('\nMetrics written to', args.csv)
