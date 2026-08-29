""" DC Deep Neural Network models """
import os

import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
import keras
from keras import layers
from keras import constraints
from keras import ops

from pvtol_model import ddy, ddz

## Curvature penalisation
# Step of the finite differences used to measure the curvature, one value per input
# (alpha, u1). The linearisation error over a box of half-widths d is bounded at second
# order by sum_i d_i^2 [H]_ii, so d must be the cross section of the tube: the values
# below are the largest deviation of the tube vertices from the linearisation trajectory
# (in alpha) and the input deviation it induces through the feedback gains (in u1),
# both measured on the closed-loop solution of the unpenalised model.
HESS_STEP = [0.118, 1.27]

PENALISE_MODES = ['g', 'h', 'gh']


def weights_file(lam=None, activation='relu'):
    """ Weight file of the DC neural network trained with penalty weight lam
    (lam = None: model of the original decomposition, without penalty)
    """
    folder = './model_ReLU' if activation == 'relu' else './model_ELU'
    if lam is None:
        return os.path.join(folder, 'f_DC.weights.h5')
    return os.path.join(folder, 'f_DC_lam{:g}.weights.h5'.format(lam))


def hessian_penalty(model, x, step):
    """ Second order measure of the curvature of a network over a box of half-widths step

    The second derivative is evaluated by central finite differences,

        sum_i [ f(x + d_i e_i) - 2 f(x) + f(x - d_i e_i) ]  =  sum_i d_i^2 [H]_ii,

    where e_i is the i-th standard basis vector of the input space. This is the second
    order term of the linearisation error over the box, and is non-negative for a convex
    f. The differences are not divided by d_i^2: each input direction has to be weighted
    by the tube cross section, otherwise the sum is governed by the input with the largest
    second derivative rather than by the one contributing most of the error.

    A ReLU network is piecewise affine, so its exact Hessian is zero almost everywhere and
    cannot be used here; the finite step measures instead the gradient jumps met within a
    window of that size. The expression is a combination of forward passes and is
    therefore differentiable with respect to the weights.

    Input: network, batch of inputs x (batch, n_in), steps d (n_in,)
    Output: scalar penalty
    """
    n_in = x.shape[-1]
    step = np.atleast_1d(np.asarray(step, dtype='float32'))
    if step.size == 1:
        step = np.repeat(step, n_in)
    f0 = model(x)
    pen = 0.
    eye = ops.eye(n_in, dtype=x.dtype)
    for i in range(n_in):
        e = float(step[i])*eye[i]
        pen = pen + ops.sum(ops.mean(model(x + e) - 2.*f0 + model(x - e), axis=0))
    return pen


class HessianPenalty(layers.Layer):
    """ Layer adding lam times the curvature of the penalised networks to the loss

    The layer leaves the output of the model unchanged and only contributes to the
    training loss, so that the decomposition used online is the one obtained from the
    weights, unmodified.
    """
    def __init__(self, lam, models, step, **kwargs):
        super().__init__(**kwargs)
        self.lam = lam
        self.step = step
        self._models = models

    def call(self, inputs):
        x, output = inputs
        pen = 0.
        for model in self._models:
            pen = pen + hessian_penalty(model, x, self.step)
        self.add_loss(self.lam*pen)
        return output


def grad_predict(x, weights):
    """ Gradient of a convex_NN model from its weights (ReLU activation)

    Input: evaluation points x (n_in, P), weights
    Output: gradients (n_out, n_in, P)
    """
    x = np.atleast_2d(x)
    x0 = x
    W = weights[0].T
    b = weights[1].T
    z = W @ x + b[:, None]
    a = np.maximum(z, 0)
    J = (z > 0).astype(float)[:, None, :] * W[:, :, None]

    N = (len(weights)-4)//4
    for i in range(N):
        Wx = weights[2+i*4].T
        bx = weights[2+i*4+1].T
        W0 = weights[2+i*4+2].T
        b0 = weights[2+i*4+3].T
        z = Wx @ a + bx[:, None] + W0 @ x0 + b0[:, None]
        a = np.maximum(z, 0)
        J = (z > 0).astype(float)[:, None, :] * (np.einsum('ij,jkp->ikp', Wx, J)
                                                 + W0[:, :, None])

    return np.einsum('oj,jkp->okp', weights[-2].T, J)


def lin_err(weights, x_0, d, sigma=lambda x: np.maximum(x, 0)):
    """ Linearisation error of a convex network, e = f(x_0 + d) - |f_0|(x_0 + d)

    Input: linearisation points x_0 (n_in, P), steps d (n_in, P) or (n_in, 1)
    Output: e (n_out, P), non-negative by convexity
    """
    d = np.broadcast_to(d, x_0.shape)
    y1 = weight_predict(x_0 + d, sigma, weights)
    y0 = weight_predict(x_0, sigma, weights)
    G = grad_predict(x_0, weights)

    return y1 - y0 - np.einsum('okp,kp->op', G, d)


def convex_NN(N_layer, N_node, sigma):
    """ Create a densely connected neural network with convex input-output map
    Input: 
        - N_layer: number of hidden layers
        - N_node: number of nodes per layer
        - sigma: activation function
    Output: neural network model
    """

    input = keras.Input(shape=(2,))
    x = input
    x = layers.Dense(N_node)(input)
    x = sigma()(x)
    
    # Add N_layer dense layers with N_node nodes
    for i in range(N_layer):
        x1 = layers.Dense(N_node, kernel_constraint=constraints.NonNeg())(x)
        #x1 = layers.LeakyReLU(alpha=0.3)(x1)
        x2 = layers.Dense(N_node)(input)
        x = layers.Add()([x1, x2])
        x = sigma()(x)
    
    output = layers.Dense(2, kernel_constraint=constraints.NonNeg())(x)
    
    return keras.Model(input, output)

def weight_predict(x, sigma, weights):
    """ 
    Model prediction from weights 
    
    """
    
    # First layer
    x0 = x
    W = weights[0].T
    b = weights[1].T
    z = W @ x + b[:, None]
    x = sigma(z)

    # Internal layers
    N = (len(weights)-4)//4
    for i in range(N):
        Wx = weights[2+i*4].T
        bx = weights[2+i*4+1].T
        W0 = weights[2+i*4+2].T
        b0 = weights[2+i*4+3].T
        
        z = Wx @ x  + bx[:, None] +  W0 @ x0 + b0[:, None]
        x = sigma(z)
    
    # Last layer
    W = weights[-2].T
    b = weights[-1].T
    z = W @ x + b[:, None]
    
    return z #sigma(z) 
    
def split(N_unit, N_layer, sigma, activation, N_batch, N_epoch,
                                                  x_train, x_test, y_train, y_test, load,
                                                  lam=0., penalise='h',
                                                  step=None, file_name=None):
    """
    Obtain DC decomposition of function f using DC neural networks

    The curvature of the decomposition can be penalised at training stage by setting
    lam > 0: a term lam*sum_i d_i^2 [H]_ii is then added to the loss (see
    hessian_penalty), for the networks selected by penalise ('g', 'h' or 'gh'). The
    linearisation error of the concave part inflates the tube of the MPC scheme, so
    reducing its curvature reduces the conservatism of the tube. The penalty applies to
    training only: the online problem is unchanged.

    Input (in addition to the arguments of the unpenalised decomposition):
        - lam: weight of the curvature penalty, 0 for no penalty
        - penalise: networks whose curvature is penalised, 'g', 'h' or 'gh'
        - step: finite difference step per input, defaults to HESS_STEP
        - file_name: weight file, defaults to the one given by weights_file
    """

    # Dimensions
    N_arg = x_train.shape[0]  # number of input to NN

    # Build model
    input = keras.Input(shape=(N_arg,))
    model_g = convex_NN(N_layer, N_unit, sigma)
    model_h = convex_NN(N_layer, N_unit, sigma)
    g = model_g(input)
    h = model_h(input)

    output = layers.Subtract()([g, h])

    if lam > 0:  # penalise the curvature of the selected networks
        penalised = {'g': [model_g], 'h': [model_h], 'gh': [model_g, model_h]}[penalise]
        output = HessianPenalty(lam, penalised,
                                HESS_STEP if step is None else step)([input, output])

    model_f_DC = keras.Model(inputs=input, outputs=output)

    # Compile
    model_f_DC.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    # Load or train model
    if file_name is None:
        file_name = weights_file(None, activation)

    if load:  # load existing model
    
        # Restore the weights
        model_f_DC.load_weights(file_name)

    else:  # train new model
        
        print("************ Training of the DC neural network... ******************")
        # Train model
        history = model_f_DC.fit(x_train.T, y_train.T, batch_size=N_batch, 
                                                     epochs=N_epoch, validation_split=0.2)
        
        # Save the weights
        model_f_DC.save_weights(file_name)
    
    # Evaluate
    test_scores = model_f_DC.evaluate(x_test.T, y_test.T, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])
    
    return model_f_DC, model_g, model_h
    


def plot(model_f_DC, model_g, model_h, sigma, param):
    """ Plot results of decomposition """

    # Generate plot data
    N_arg = model_f_DC.get_weights()[0].shape[0]
    N_test = 10
    u = np.linspace(-param.u_max, param.u_max, N_test)
    alpha = np.linspace(-param.x_max, param.x_max, N_test)
    X, U = np.meshgrid(alpha, u)
    F1_y = np.zeros_like(X)
    F2_y = np.zeros_like(X)
    F1_z = np.zeros_like(X)
    F2_z = np.zeros_like(X)
    err_y = np.zeros_like(X)
    err_z = np.zeros_like(X)
    DDY = ddy(X, U, param)
    DDZ = ddz(X, U, param)
    
    x = np.zeros((N_test**2, N_arg))
    k = 0
    for h1 in alpha:
        for h2 in u: 
            x[k, :] = np.array([h1, h2])
            k += 1
            
    y  = model_f_DC.predict(x)
    y1 = model_g.predict(x)
    y2 = model_h.predict(x)
    
    y1_ = weight_predict(x.T, sigma, model_g.get_weights())
    y2_ = weight_predict(x.T, sigma, model_h.get_weights())
    
    """print("check weight_predict")
    print(y1-y1_.T)
    print(y2-y2_.T)
    print("Max error dy1: ", np.max(y1-y1_.T))
    print("Max error dy2: ", np.max(y2-y2_.T))"""

    
    for i in range(N_test):
        for j in range(N_test):
            xu = np.vstack([X[i, j], U[i, j]])
            F1_y[i, j] = weight_predict(xu, sigma, model_g.get_weights())[0, 0]
            F2_y[i, j] = weight_predict(xu, sigma, model_h.get_weights())[0, 0]
            F1_z[i, j] = weight_predict(xu, sigma, model_g.get_weights())[1, 0]
            F2_z[i, j] = weight_predict(xu, sigma, model_h.get_weights())[1, 0]
            err_y[i, j] = np.abs(DDY[i, j] - (F1_y[i, j] - F2_y[i, j]))
            err_z[i, j] = np.abs(DDZ[i, j] - (F1_z[i, j] - F2_z[i, j]))
    
    """print("************ Error in DC approximation ****************")
    print("Max absolute error: [dy, dz] = [{}, {}]".format(err_y.max(), err_z.max()))
    print("Mean absolute error: [dy, dz] = [{}, {}]".format(err_y.mean(), err_z.mean()))"""
    fig = plt.figure(figsize=plt.figaspect(0.5))
        
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_wireframe(X, U, DDY, rstride=3, cstride=3, label='ref')
    ax.scatter(x[:,0], x[:,1], y[:, 0], label='$f=f_1-f_2$')
    ax.scatter(x[:,0], x[:,1], y1[:, 0], label='$f_1$')
    ax.scatter(x[:,0], x[:,1], y2[:, 0], label='$f_2$')
    ax.set_xlabel('alpha')
    ax.set_ylabel('$u_1$')
    ax.set_zlabel('$\ddot{y}$')
    ax.legend()
        
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_wireframe(X, U, DDZ, rstride=3, cstride=3, label='ref')
    ax.scatter(x[:,0], x[:,1], y[:, 1], '-r', label='$f=f_1-f_2$')
    ax.scatter(x[:,0], x[:,1], y1[:, 1], label='$f_1$')
    ax.scatter(x[:,0], x[:,1], y2[:, 1], label='$f_2$')
    ax.set_xlabel('alpha')
    ax.set_ylabel('$u_1$')
    ax.set_zlabel('$\ddot{z}$')
    ax.legend()
    
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    c0 = ax.plot_surface(X, U, F1_y-F2_y, alpha=0.7, linewidth=0, 
                                              antialiased=True, shade=True, label='g - h')
    ax.scatter(X.flatten(), U.flatten(), DDY.flatten(), label='data')
    c1 = ax.plot_surface(X, U, F1_y, alpha=0.7, linewidth=0, 
                                                  antialiased=True, shade=True, label='g')
    c2 = ax.plot_surface(X, U, F2_y, alpha=0.7, linewidth=0, 
                                                  antialiased=True, shade=True, label='h')
    ax.set_xlabel('alpha')
    ax.set_ylabel('$u_1$')
    ax.set_zlabel('$\ddot{y}$')
    c0._facecolors2d = c0._facecolor3d
    c0_edgecolors2d = c0._edgecolor3d
    c1._facecolors2d = c1._facecolor3d
    c1._edgecolors2d = c1._edgecolor3d
    c2._facecolors2d = c2._facecolor3d
    c2._edgecolors2d = c2._edgecolor3d
    ax.legend()
    
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    
    c0 = ax.plot_surface(X, U, F1_z-F2_z, alpha=0.7, linewidth=0, 
                                              antialiased=True, shade=True, label='g - h')
    
    ax.scatter(X.flatten(), U.flatten(), DDZ.flatten(), label='data')
    
    c1 = ax.plot_surface(X, U, F1_z, alpha=0.7, linewidth=0, 
                                                  antialiased=True, shade=True, label='g')
    
    c2 = ax.plot_surface(X, U, F2_z, alpha=0.7, linewidth=0, 
                                                  antialiased=True, shade=True, label='h')
    ax.set_xlabel('alpha')
    ax.set_ylabel('$u_1$')
    ax.set_zlabel('$\ddot{z}$')
    c0._facecolors2d = c0._facecolor3d
    c0_edgecolors2d = c0._edgecolor3d
    c1._facecolors2d = c1._facecolor3d
    c1._edgecolors2d = c1._edgecolor3d
    c2._facecolors2d = c2._facecolor3d
    c2._edgecolors2d = c2._edgecolor3d
    ax.legend()
    
    
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1)
    cs = ax.contourf(X, U, err_y, cmap='viridis') #locator=ticker.LogLocator()
    ax.set_xlabel('alpha')
    ax.set_ylabel('$u_1$')
    ax.set_title('y-axis least-squares absolute error [$m / s^{-2}$]')
    fig.colorbar(cs)
    
    ax = fig.add_subplot(1, 2, 2)
    cs = ax.contourf(X, U, err_z, cmap='viridis') #locator=ticker.LogLocator()
    ax.set_xlabel('alpha')
    ax.set_ylabel('$u_1$')
    ax.set_title('z-axis least-squares absolute error [$m / s^{-2}$]')
    fig.colorbar(cs)
          
    plt.show()
    
    # Graph
    keras.utils.plot_model(model_f_DC, "f_DC.png", show_shapes=True)
    keras.utils.plot_model(model_g, "f1.png", show_shapes=True)
    
    """print("Weights: ")
    for w in model_g.get_weights():
        print("new w: ")
        print(w)"""

## Hessian
def D_2(f, x_0, delta, i, j):
    """ 
    Evaluate second derivative of f along x_i and x_j at x_0:
    D_2 f = d^2 f /dx_i dx_j
    
    Input: function to differentiate f, evaluation point x_0, step delta, 
    indices of variables along which to differentiate i and j.
    Output: second order derivative along x_i and x_j
    """
    n = len(x_0)
    I = np.eye(n)
    
    return (f(x_0 + delta*I[j, :] + delta*I[i, :]) -f(x_0 + delta*I[j, :])
            - f(x_0 + delta*I[i, :]) + f(x_0))/delta**2

def hess(f, x_0, delta):
    """
    Evaluate the Hessian of f at x_0 (numerically)
    
    Input: function whose Hessian is to be computed f, evaluation point x_0, 
    differentiation step delta. 
    Output: Hessian H. 
    """
    n = len(x_0)
    H = np.empty((n,n))
    
    for i in range(n):
        for j in range(n):
            H[i, j] = D_2(f, x_0, delta, i, j)  # compute 2nd derivative along x_i and x_j
    
    return H
    
## Check split
def check(f, g, h, x, p):
    """ A function to check the validity of a given DC decomposition
    
    f = g - h where g and h are convex
    
    Will perform a series of checks to assess: 
    - if the DC decomposition describes well the original function f
    - if g, h are convex
    
    Input: 
        - f: original function
        - g, h: convex functions of the DC decomposition of f
        - x: test points
        - p: structure of parameters
    
    Output: None
    """
    
    ## 1. Check f = g-h
    N = x.shape[1]  # number of test points
    
    # Compute the error of DC decomposition
    err_split = np.abs(g(x)-h(x)-f(*x, p))
    
    #print("************ Errors in LS approximation ****************")
    #print("Max sample Fs: ", np.abs(F_s).max(), "/ Max absolute error: ", err_LS.max())
    #print("Mean sample Fs: ", np.abs(F_s).mean(), "/ Mean absolute error: ",err_LS.mean())
    
    print("************ Error in DC approximation ****************")
    print("Mean absolute error [dy_mean dz_mean] = ", err_split.mean(axis=1))
    print("Max absolute error [dy_max dz_max] = ", err_split.max(axis=1))
    
    ## 2. Check convexity of g and h
    # Define functions 
    g1 = lambda x: g(x)[0, 0]
    g2 = lambda x: g(x)[1, 0]
    h1 = lambda x: h(x)[0, 0]
    h2 = lambda x: h(x)[1, 0]
    
    print("********** Checking convexity of g and h **************")
    viol = 0
    tol = .01     # tolerance for Hessian eigenvalues non-negativity
    delta = .001  # step for 2nd order derivative computation
    for i in range(N):
        # Hessian functions
        Hfun_g1 = nd.Hessian(g1)
        Hfun_h1 = nd.Hessian(h1)
        Hfun_g2 = nd.Hessian(g2)
        Hfun_h2 = nd.Hessian(h2)
        
        # Evaluate Hessians at test point
        """H_g1 = Hfun_g1(x[:, i])
        H_h1 = Hfun_h1(x[:, i])
        H_g2 = Hfun_g2(x[:, i])
        H_h2 = Hfun_h2(x[:, i])"""
        H_g1 = hess(g1, x[:, i], delta)
        H_h1 = hess(h1, x[:, i], delta)
        H_g2 = hess(g2, x[:, i], delta)
        H_h2 = hess(h2, x[:, i], delta)
        
        # Compute eigenvalues
        eig_g1 = np.linalg.eigvals(H_g1)
        eig_h1 = np.linalg.eigvals(H_h1)
        eig_g2 = np.linalg.eigvals(H_g2)
        eig_h2 = np.linalg.eigvals(H_h2)
        
        # Stack all eigenvalues
        eig_all = np.stack([eig_g1, eig_h1, eig_g2, eig_h2])
        #print("Eigen values: ")
        #print(eig_all)
        
        # Check if any eigenvalue is negative (up to a given tolerance)
        if np.any(eig_all < -tol):
            print("Hessian not psd at iteration", i, "in x: ", x[:, i], "\n")
            print("Eigenvalues: ", eig_all)
            viol += 1
        
        
        """# Check PSDness of Hessians (will raise 'LinAlgError' exception if not PSD)
        try: 
            scipy.linalg.cholesky(H_g1)
            scipy.linalg.cholesky(H_h1)
            scipy.linalg.cholesky(H_g2)
            scipy.linalg.cholesky(H_h2)
        except np.linalg.LinAlgError:
            print("Hessian not psd at iteration", i, "in x: ", x[i, :], "\n")
            viol += 1"""
    
    print("Checking done.")
    
    if viol == 0: print("No convexity violations.")
    else: print("{} convexity violations detected !".format(viol))