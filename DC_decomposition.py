""" DC Deep Neural Network models """
import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
import keras
from keras import layers
from keras.constraints import NonNeg

from pvtol_model import ddy, ddz


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
        x1 = layers.Dense(N_node, kernel_constraint=NonNeg())(x)
        #x1 = layers.LeakyReLU(alpha=0.3)(x1)
        x2 = layers.Dense(N_node)(input)
        x = layers.Add()([x1, x2])
        x = sigma()(x)
    
    output = layers.Dense(2, kernel_constraint=NonNeg())(x)
    
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
                                                  x_train, x_test, y_train, y_test, load):
    """ 
    Obtain DC decomposition of function f using DC neural networks 
    
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
    
    model_f_DC = keras.Model(inputs=input, outputs=output)

    # Compile 
    model_f_DC.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    
    # Load or train model
    if activation == "relu": 
        file_name = './model_ReLU/f_DC.weights.h5'
    elif activation == "elu": 
        file_name = './model_ELU/f_DC.weights.h5'
    
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
    
    return (f(x_0 + delta*I[j, :] + delta*I[i, :]) -f(x_0 + delta*I[j, :])\
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