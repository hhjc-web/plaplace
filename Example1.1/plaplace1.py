# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 17:04:08 2024

@author: USUARIO
"""

import numpy as np
import tensorflow as tf
import math
from math import ceil, floor
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow import convert_to_tensor
from tensorflow.keras.optimizers import Adam
from scipy.optimize import minimize
from scipy.linalg import cholesky, LinAlgError
from scipy import sparse
import json
import os
import scipy.io as sio

def load_hparams(file_config):
    with open(file_config, 'r') as file:
        config = json.load(file)
    return config


config = load_hparams('config.json')

seed = config["seed"]["seed"]  # Seed

# --------------- gradnet Architecture hyperparameters ----------------------------------
grad_neurons = config["grad_architecture_hparams"]["neurons"]  # Neurons in every hidden layer
grad_layers = config["grad_architecture_hparams"]["layers"]  # Hidden layers
grad_output_dim = config["grad_architecture_hparams"]["output_dim"]  # Output dimension
# -------------------------------------------------------------------------------

# -------------- Adam hyperparameters -------------------------------------------
Adam_epochs = config["Adam1_hparams"]["Adam_epochs"]  # Adam epochs
lr0 = config["Adam1_hparams"]["lr0"]  # Initial learning rate (We consider an exponential lr schedule)
decay_steps = config["Adam1_hparams"]["decay_steps"]  # Decay steps
decay_rate = config["Adam1_hparams"]["decay_rate"]  # Decay rate
b1 = config["Adam1_hparams"]["b1"]  # beta1
b2 = config["Adam1_hparams"]["b2"]  # beta2
epsilon = config["Adam1_hparams"]["epsilon"]  # epsilon
Nprint_adam = config["Adam1_hparams"]["Nprint_adam"]  # Adam results will be printed and save every Nprint_adam iters
# ------------------------------------------------------------------------------

# ------------ Batch hyperparameters -------------------------------------------
Nint = config["batch_hparams"]["Nint"]  # Number of points at batch
Nbdy = config["batch_hparams"]["Nbdy"]  # Number of points at batch
Nchange = config["batch_hparams"]["Nchange"]  # Batch is changed every Nchange iterations
k1 = config["batch_hparams"]["k1"]  # k hyperparameter (see adaptive_rad function below)
k2 = config["batch_hparams"]["k2"]  # c hyperparameter (see adaptive_rad function below)
# ------------------------------------------------------------------------------

# ------------ problem hyperparameters -------------------------------------------
dim = 3
domain_size = 4/3*math.pi
bdy_size = 4*math.pi
p = config["problem"]["p"]
q = p/(p-1)
# ------------------------------------------------------------------------------

# ------------ Test hyperparameters --------------------------------------
Nx = config["test_hparams"]["Nx"]  # Number of grid points for test set in x direction
Ny = config["test_hparams"]["Ny"]  # Number of grid points for test set in y direction

# ------------ Quasi-Newton (QN) hyperparameters -------------------------------
Nbfgs = config["bfgs_hparams"]["BFGS_epochs"]  # Number of QN iterations
method = config["bfgs_hparams"]["method"]  # Method. See below
method_bfgs = config["bfgs_hparams"]["method_bfgs"]  # Quasi-Newton algorithm. See below
use_sqrt = config["bfgs_hparams"]["use_sqrt"]  # Use square root of the MSE loss to train
use_log = config["bfgs_hparams"]["use_log"]  # Use log of the MSE loss to train
Nprint_bfgs = config["bfgs_hparams"]["Nprint_bfgs"]  # QN results will be printed and save every Nprint_adam iters

# In method, you can choose between:
# -BFGS: Here, we include BFGS and the different self-scaled QN methods.
#        To distinguish between these algorithms, we use method_bfgs. See below
# -bfgsr: Personal implementation of the factored BFGS Hessian approximations.
#         See https://ccom.ucsd.edu/reports/UCSD-CCoM-22-01.pdf for details
#         Very slow, to be optimized.
# -bfgsz: Personal implementation of the factored inverse BFGS Hessian approximations.
#         See https://ccom.ucsd.edu/reports/UCSD-CCoM-22-01.pdf for details
#         Comparable with BFGS in terms of speed.

# If method=BFGS, the variable "method_bfgs" chooses the different QN methods.
# The options for this are (see the modified Scipy optimize script):
# -BFGS_scipy: The original implementation of BFGS of Scipy
# -BFGS: Equivalent implementation, but faster (avoid repeated calculations in the BFGS formula)
# -SSBFGS_AB: The Self-scaled BFGS formula, where the tauk coefficient is calculated with
#            Al-Baali's formula (Formula 11 of "Unveiling the optimization process in PINNs")
# -SSBFGS_OL Same, but tauk is calculated with the original choice of Oren and Luenberger (not recommended)
# -SSBroyden2: Here we use the tauk and phik expressions defined in the paper
#             (Formulas 13-23 of "Unveiling the optimization process in PINNs")
# -SSbroyden1: Another possible choice for these parameters (sometimes better, sometimes worse than SSBroyden1)

# ------------------------------------------------------------------------------

tf.keras.backend.set_floatx('float64')
tf.get_logger().setLevel('ERROR')
tf.keras.utils.set_random_seed(seed)

def generate_model(layer_dims):
    # input (x, y)
    X_input = Input((layer_dims[0],))

    # first hidden layer
    X = Dense(layer_dims[1], activation="tanh")(X_input)

    # hidden layer
    for i in range(2, len(layer_dims) - 1):
        X = Dense(layer_dims[i], activation="tanh")(X)

    # output layer
    X = Dense(layer_dims[-1], activation=None)(X)

    return Model(inputs=X_input, outputs=X)

def generate_inputs(Nint):
    r = tf.random.uniform([Nint, 1], minval=0, maxval=1, dtype=tf.float64)
    mean = np.zeros([dim])
    cov = np.eye(dim)
    theta_np = np.random.multivariate_normal(mean, cov, Nint)
    theta = convert_to_tensor(theta_np)
    theta_norm = tf.norm(theta, axis=1, keepdims=True)
    theta = theta / theta_norm
    result = r**(1.0 / dim) * theta
    return result

def generate_test(Nx, Ny):
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, Ny)
    x, y = np.meshgrid(x, y)
    X = np.hstack((x.flatten()[:, None], y.flatten()[:, None]))
    return X, x, y

def solution(X):
    r = tf.reduce_sum(X ** 2, axis=1, keepdims=True) ** (1 / 2)
    result = 1 - r ** (p / (p - 1))
    return result

def generate_bdy(Nbdy, dim, domain_size, bdy_size):
    mean = np.zeros([dim])
    cov = np.eye(dim)
    theta_np = np.random.multivariate_normal(mean, cov, Nbdy)
    theta = tf.constant(theta_np, dtype=tf.float64)
    result = theta / tf.sqrt(tf.reduce_sum(theta ** 2, axis=1, keepdims=True))

    # x_domain = generate_inputs(10000)
    # r = tf.reduce_sum(x_domain ** 2, axis=1, keepdims = True)**(1/2)
    # f = 3 * (p/(p-1))**(p-1)+r-r
    # un_bdy = 1/bdy_size * domain_size * tf.reduce_mean(f) + result[:,0:1] - result[:,0:1]

    return convert_to_tensor(result)

def adaptive_rad(GRAD, Nint, rad_args, Ntest=100000):
    '''
    Parameters
    ----------
    N : TENSORFLOW MODEL
        DESCRIPTION.
        PINN model, obtained with generate_model() function

    Nint : INTEGER
           DESCRIPTION.
           Number of training points in a given batch

    rad_args: TUPLE (k1,k2)
              DESCRIPTION.
              Adaptive resampling of Wu et al. (2023), formula (2)
              k1: k
              k2: c
              DOI: https://doi.org/10.1016/j.cma.2022.115671

    Ntest: INTEGER
           DESCRIPTION.
           Number of test points to do the resampling

    Returns
    -------
    X: Batch of points (in Tensorflow format)
    TYPE
        DESCRIPTION.
    '''
    Xtest = generate_inputs(Ntest)
    k1, k2 = rad_args
    Y = tf.math.abs(get_results(GRAD, Xtest)[-1]).numpy()
    err_eq = np.power(Y, k1) / np.power(Y, k1).mean() + k2
    err_eq_normalized = (err_eq / sum(err_eq))
    X_ids = np.random.choice(a=len(Xtest), size=Nint, replace=False,
                             p=err_eq_normalized)
    return tf.gather(Xtest, X_ids)

def output(model, X):
    '''
    Parameters
    ----------
    N : TENSORFLOW MODEL
        PINN model, obtained with generate_model() function
    X : TENSOR
        Batch of points (in Tensorflow format)
    Returns
    -------
    u: TENSOR
        PINN prediction. Fourier

    '''
    Nout = model(X)
    u = Nout[:, 0, None]

    return u

def get_results(GRAD, X):
    global p,q
    '''
    Parameters
    ----------
    GRAD : TENSORFLOW MODEL
        PINN model, obtained with generate_model() function
    X : TENSOR
        Batch of points (in Tensorflow format)

    Returns
    -------
    u : TENSOR
        PINN prediction
    fu : TENSOR
         PDE residuals

    '''
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]

    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt1:
        gt1.watch(X)

        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt2:
            gt2.watch(X)

            u = output(GRAD, X)

        ugrad = gt2.gradient(u, X)
        u_x = ugrad[:, 0]
        u_y = ugrad[:, 1]
        u_z = ugrad[:, 2]

    u_xx = gt1.gradient(u_x, X)[:, 0]
    u_yy = gt1.gradient(u_y, X)[:, 1]
    u_zz = gt1.gradient(u_z, X)[:, 2]

    r = tf.sqrt(x**2+y**2+z**2)
    f = 3 * (p / (p - 1)) ** (p - 1) + r - r
    fu = u_xx + u_yy + u_zz - f

    return u, fu

def bdy_loss(GRAD, X_bdy):
    global q
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt1:
        gt1.watch(X_bdy)
        u = output(GRAD, X_bdy)

    ugrad = gt1.gradient(u, X_bdy)
    outernormal = X_bdy
    # un = tf.reduce_sum(ugrad * outernormal, axis=1, keepdims=True)
    # X_unique = 0*X_bdy[0:1,:]
    # return tf.reduce_mean(tf.math.abs(un - un_bdy)**q) + tf.reduce_mean(output(GRAD, X_unique))**2
    utan = ugrad - tf.reduce_sum(ugrad * outernormal, axis=1, keepdims=True) * outernormal

    return tf.reduce_mean(tf.math.abs(u) ** q) ** (1/q) + tf.reduce_mean(tf.math.abs(utan) ** q) ** (1/q)

def total_loss(GRAD, X_domain, X_bdy, lambda_bdy):
    global p,q
    _, f_residual = get_results(GRAD, X_domain)
    loss_pde = tf.reduce_mean(tf.math.abs(f_residual)**q) ** (1/q)
    loss_bdy = bdy_loss(GRAD, X_bdy)
    return loss_pde + lambda_bdy * loss_bdy

@tf.function  # Precompile training function to accelerate the process
def traininggrad(GRAD, X_domain, X_bdy, optimizer, lambda_bdy):
    with tf.GradientTape() as tape:
        loss_value = total_loss(GRAD, X_domain, X_bdy, lambda_bdy)
    grads = tape.gradient(loss_value, GRAD.trainable_variables)
    optimizer.apply_gradients(zip(grads, GRAD.trainable_variables))
    return loss_value

rad_args = (k1, k2)
epochs = np.arange(Nprint_adam, Adam_epochs + Nprint_adam, Nprint_adam)
loss_list = np.zeros(len(epochs))  # loss list
X = generate_inputs(Nint)

grad_layer_dims = [None] * (grad_layers + 2)
grad_layer_dims[0] = X.shape[1]
for i in range(1, len(grad_layer_dims)):
    grad_layer_dims[i] = grad_neurons
grad_layer_dims[-1] = grad_output_dim
GRAD = generate_model(grad_layer_dims)

lr = tf.keras.optimizers.schedules.ExponentialDecay(lr0, decay_steps, decay_rate)

optimizer_grad = Adam(lr, b1, b2, epsilon=epsilon)
template = 'Epoch {}, loss: {}'
X_bdy = generate_bdy(Nbdy, dim, domain_size, bdy_size)
lambda_bdy = 20

# ------------------- Adam TRAINING LOOP ---------------------------------------
for i in range(Adam_epochs):
    if (i + 1) % Nchange == 0:
        X = adaptive_rad(GRAD, Nint, rad_args)
        # X = random_permutation(X)
    if (i + 1) % Nprint_adam == 0:
        loss_value = total_loss(GRAD, X, X_bdy, lambda_bdy)
        print("i=", i + 1)
        print(template.format(i + 1, loss_value))
        loss_list[i // Nprint_adam] = loss_value.numpy()

    traininggrad(GRAD, X, X_bdy, optimizer_grad, lambda_bdy)

np.savetxt(f"loss_adam_first.txt", np.c_[epochs, loss_list])
initial_weights = np.concatenate([tf.reshape(w, [-1]).numpy() \
                                  for w in GRAD.weights])  # initial set of trainable variables
grad_layer_dims[0] = dim

def nested_tensor(tparams, grad_layer_dims):
    '''

    Parameters
    ----------
    tparams : NUMPY ARRAY
        DESCRIPTION: Trainable parameters in Numpy array format
    grad_layer_dims : TUPLE
        DESCRIPTION:
        (L0,L1,...,Ln), where Li is the number of neurons at ith layer
                        if i=0, Li corresponds to the input dimension

    Returns
    -------
    temp : LIST
           List of tensors (Trainable variables in Tensorflow format)

    '''
    temp = [None] * (2 * len(grad_layer_dims) - 2)
    index = 0
    for i in range(len(temp)):
        if i % 2 == 0:
            temp[i] = np.reshape(tparams[index:index + grad_layer_dims[i // 2] * \
                                               grad_layer_dims[i // 2 + 1]], (grad_layer_dims[i // 2],
                                                                         grad_layer_dims[i // 2 + 1]))
            index += grad_layer_dims[i // 2] * grad_layer_dims[i // 2 + 1]
        else:
            temp[i] = tparams[index:index + grad_layer_dims[i - i // 2]]
            index += grad_layer_dims[i - i // 2]
    return temp

@tf.function
def loss_and_gradient_TF(GRAD, X, X_bdy, use_sqrt, use_log, lambda_bdy):
    with tf.GradientTape() as tape:
        total = total_loss(GRAD, X, X_bdy, lambda_bdy)

        if use_sqrt:
            loss_value = tf.sqrt(total)
        elif use_log:
            loss_value = tf.math.log(total)
        else:
            loss_value = total
    gradsN = tape.gradient(loss_value, GRAD.trainable_variables)
    return loss_value, gradsN


# LOSS AND GRADIENT IN NUMPY FORMAT
def loss_and_gradient(weights, GRAD, X, X_bdy, grad_layer_dims, use_sqrt, use_log, lambda_bdy):
    resh_weights = nested_tensor(weights, grad_layer_dims)
    GRAD.set_weights(resh_weights)
    loss_value, grads = loss_and_gradient_TF(GRAD, X, X_bdy, use_sqrt, use_log, lambda_bdy)
    grads_flat = np.concatenate([
        tf.reshape(g, [-1]).numpy() if g is not None else np.zeros_like(w.numpy().reshape(-1))
        for g, w in zip(grads, GRAD.trainable_variables)
    ])
    return loss_value.numpy(), grads_flat


# ------------------------- BFGS TRAINING --------------------------------------
epochs_bfgs = np.arange(0, Nbfgs + Nprint_bfgs, Nprint_bfgs)  # iterations bfgs list
epochs_bfgs += Adam_epochs
lossbfgs = np.zeros(len(epochs_bfgs))  # loss bfgs list
error_list = np.zeros(len(epochs_bfgs))  # loss bfgs list
cont = 0

# Xtest, x, y = generate_test(Nx, Ny)
# Xtest = np.hstack((Xtest, 0*Xtest[:,0:1]))
# Xtest = convert_to_tensor(Xtest)
# uexact = solution(Xtest)
# r = tf.reduce_sum(Xtest**2, axis = 1, keepdims = True)**(1/2)
# uexact = tf.where(r<1, uexact, 0)
# uexact = uexact.numpy()

def callback(*, intermediate_result):
    global GRAD, cont, lossbfgs, Nprint_bfgs
    if (cont + 1) % Nprint_bfgs == 0 or cont == 0:
        if use_sqrt:
            loss_value = intermediate_result.fun
        elif use_log:
            loss_value = np.exp(intermediate_result.fun)
        else:
            loss_value = intermediate_result.fun
        lossbfgs[(cont + 1) // Nprint_bfgs] = loss_value

        # utest = output(GRAD, Xtest)
        # r = tf.reduce_sum(Xtest ** 2, axis=1, keepdims=True) ** (1 / 2)
        # utest = tf.where(r < 1, utest, 0)
        # utest = utest.numpy()
        # error = np.linalg.norm(utest - uexact) / np.linalg.norm(uexact)
        # error_list[(cont + 1) // Nprint_bfgs] = error

        print(loss_value, cont + 1)
    cont = cont + 1

if method == "BFGS":
    method_bfgs = method_bfgs
    initial_scale = False
    H0 = tf.eye(len(initial_weights), dtype=tf.float64)
    H0 = H0.numpy()
    options = {'maxiter': Nchange, 'gtol': 0, "hess_inv0": H0,
               "method_bfgs": method_bfgs, "initial_scale": initial_scale}

elif method == "bfgsr":
    R0 = sparse.csr_matrix(np.eye(len(initial_weights)))
    options = {"maxiter": Nchange, "gtol": 0, "r_inv0": R0}

elif method == "bfgsz":
    Z0 = tf.eye(len(initial_weights), dtype=tf.float64)
    Z0 = Z0.numpy()
    options = {"maxiter": Nchange, "gtol": 0, "Z0": Z0}

while cont < Nbfgs:  # Training loop
    result = minimize(loss_and_gradient, initial_weights,
                      args=(GRAD, X, X_bdy, grad_layer_dims, use_sqrt, use_log, lambda_bdy),
                      method=method, jac=True, options=options,
                      tol=0, callback=callback)
    initial_weights = result.x

    if method == "BFGS":
        H0 = result.hess_inv
        H0 = (H0 + np.transpose(H0)) / 2
        try:
            cholesky(H0)
        except LinAlgError:
            H0 = tf.eye(len(initial_weights), dtype=tf.float64)
            H0 = H0.numpy()

        options = {'maxiter': Nchange, 'gtol': 0, "hess_inv0": H0,
                   "method_bfgs": method_bfgs, "initial_scale": initial_scale}

    elif method == "bfgsr":
        R0 = result.r_inv
        options = {"maxiter": Nchange, "gtol": 0, "r_inv0": R0}

    elif method == "bfgsz":
        Z0 = result.Z
        options = {"maxiter": Nchange, "gtol": 0, "Z0": Z0}

    X = adaptive_rad(GRAD, Nint, rad_args)

fname_loss = f"loss_bfgs_first.txt"
# fname_error = f"error_bfgs_first.txt"

np.savetxt(fname_loss, np.c_[epochs_bfgs, lossbfgs])
# np.savetxt(fname_error, np.c_[epochs_bfgs, error_list])

all_epochs = np.concatenate([epochs, epochs_bfgs])
all_losses = np.concatenate([loss_list, lossbfgs])
output_path = '.'
loss_path = os.path.join(output_path, "loss1.mat")
sio.savemat(loss_path, {"iteration": all_epochs, "solution_loss": all_losses})

GRAD.save('GRAD.keras')

#  -------------------------  first training end  -------------------------