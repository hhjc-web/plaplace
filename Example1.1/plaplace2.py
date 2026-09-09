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
from tensorflow.keras.models import Model, load_model
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

# --------------- rotnet Architecture hyperparameters ----------------------------------
rot_neurons = config["rot_architecture_hparams"]["neurons"]  # Neurons in every hidden layer
rot_layers = config["rot_architecture_hparams"]["layers"]  # Hidden layers
rot_output_dim = config["rot_architecture_hparams"]["output_dim"]  # Output dimension
# -------------------------------------------------------------------------------

# -------------- Adam hyperparameters -------------------------------------------
Adam_epochs = config["Adam2_hparams"]["Adam_epochs"]  # Adam epochs
lr0 = config["Adam2_hparams"]["lr0"]  # Initial learning rate (We consider an exponential lr schedule)
decay_steps = config["Adam2_hparams"]["decay_steps"]  # Decay steps
decay_rate = config["Adam2_hparams"]["decay_rate"]  # Decay rate
b1 = config["Adam2_hparams"]["b1"]  # beta1
b2 = config["Adam2_hparams"]["b2"]  # beta2
epsilon = config["Adam2_hparams"]["epsilon"]  # epsilon
Nprint_adam = config["Adam2_hparams"]["Nprint_adam"]  # Adam results will be printed and save every Nprint_adam iters
# ------------------------------------------------------------------------------

# ------------ Batch hyperparameters -------------------------------------------
Nint = config["batch_hparams"]["Nint"]  # Number of points at batch
Nbdy = config["batch_hparams"]["Nbdy"]  # Number of points at batch
Nchange = config["batch_hparams"]["Nchange"]  # Batch is changed every Nchange iterations
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
# Nbfgs = config["bfgs_hparams"]["BFGS_epochs"]  # Number of QN iterations
# method = config["bfgs_hparams"]["method"]  # Method. See below
# method_bfgs = config["bfgs_hparams"]["method_bfgs"]  # Quasi-Newton algorithm. See below
# use_sqrt = config["bfgs_hparams"]["use_sqrt"]  # Use square root of the MSE loss to train
# use_log = config["bfgs_hparams"]["use_log"]  # Use log of the MSE loss to train
# Nprint_bfgs = config["bfgs_hparams"]["Nprint_bfgs"]  # QN results will be printed and save every Nprint_adam iters

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

GRAD = load_model('GRAD.keras')
GRAD.trainable = False

def generate_model(layer_dims):
    # input (x, y)
    X_input = Input((layer_dims[0],))

    # first hidden layer
    X = Dense(layer_dims[1], activation="tanh")(X_input)

    # hidden layer
    for i in range(2, len(layer_dims) - 1):
        X = Dense(layer_dims[i], activation="tanh")(X)

    # output layer
    # X = Dense(layer_dims[-1], activation=None, use_bias=False)(X)
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
    result = r**(1.0 / dim)  * theta
    return result

def generate_test(Nx, Ny):
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, Ny)
    x, y = np.meshgrid(x, y)
    X = np.hstack((x.flatten()[:, None], y.flatten()[:, None]))
    return X, x, y

def solution(X):
    r = tf.reduce_sum(X**2, axis = 1, keepdims = True)**(1/2)
    result = 1-r**(p/(p-1))
    return result

def gradsolution(X):

    r = tf.reduce_sum(X ** 2, axis=1, keepdims=True) ** (1 / 2)
    ugrad = - p / (p - 1) * r ** ((2 - p) / (p - 1)) * X
    result = -tf.reduce_sum(ugrad**2, axis = 1, keepdims = True)**((p-2)/2) * ugrad

    return result

def generate_bdy(Nbdy, dim, domain_size, bdy_size):
    mean = np.zeros([dim])
    cov = np.eye(dim)
    theta_np = np.random.multivariate_normal(mean, cov, Nbdy)
    theta = tf.constant(theta_np, dtype=tf.float64)
    result = theta / tf.sqrt(tf.reduce_sum(theta ** 2, axis=1, keepdims=True))
    u_bdy = solution(result)

    return convert_to_tensor(result), convert_to_tensor(u_bdy)

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

    return Nout

epochs = np.arange(Nprint_adam, Adam_epochs + Nprint_adam, Nprint_adam)
loss_list = np.zeros(len(epochs))  # loss list


rot_layer_dims = [None] * (rot_layers + 2)
rot_layer_dims[0] = dim
for i in range(1, len(rot_layer_dims)):
    rot_layer_dims[i] = rot_neurons
rot_layer_dims[-1] = rot_output_dim
ROT = generate_model(rot_layer_dims)

lr = tf.keras.optimizers.schedules.ExponentialDecay(lr0, decay_steps, decay_rate)

optimizer_rot = Adam(lr, b1, b2, epsilon=epsilon)
template = 'Epoch {}, loss: {}, error: {}'
X_bdy, u_bdy = generate_bdy(Nbdy, dim, domain_size, bdy_size)

epochs = np.arange(Nprint_adam, Adam_epochs + Nprint_adam, Nprint_adam)
loss_list = np.zeros(len(epochs))  # loss list
error_list = np.zeros(len(epochs))  # loss list

Xtest, x, y = generate_test(Nx, Ny)
Xtest = np.hstack((Xtest, 0*Xtest[:,0:1]))
Xtest = convert_to_tensor(Xtest)
uexact = gradsolution(Xtest)
r = tf.reduce_sum(Xtest**2, axis = 1, keepdims = True)**(1/2)
uexact = tf.where(r<=1, uexact, 0)
uexact = uexact.numpy()

def loss(GRAD, ROT, X_domain, X_bdy, u_bdy, domain_size, bdy_size):
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt_u:
        gt_u.watch(X_domain)
        u = output(GRAD, X_domain)
    ugrad = gt_u.gradient(u, X_domain)

    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt_v1:
        gt_v1.watch(X_domain)
        v1 = output(ROT, X_domain)[:, 0:1]
    v1grad = gt_v1.gradient(v1, X_domain)

    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt_v2:
        gt_v2.watch(X_domain)
        v2 = output(ROT, X_domain)[:, 1:2]
    v2grad = gt_v2.gradient(v2, X_domain)

    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt_v3:
        gt_v3.watch(X_domain)
        v3 = output(ROT, X_domain)[:, 2:3]
    v3grad = gt_v3.gradient(v3, X_domain)

    vrot = tf.concat([v3grad[:, 1:2] - v2grad[:, 2:3], v1grad[:, 2:3] - v3grad[:, 0:1], v2grad[:, 0:1] - v1grad[:, 1:2]], axis=1)
    sigma = ugrad + vrot
    sigmastar = gradsolution(X_domain)
    # print(tf.norm(sigma-sigmastar)/tf.norm(sigmastar))
    loss_pde = domain_size / q * tf.reduce_mean(tf.reduce_sum(sigma**2, axis = 1, keepdims = True)**(q/2))

    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt_u:
        gt_u.watch(X_bdy)
        u = output(GRAD, X_bdy)
    ugrad = gt_u.gradient(u, X_bdy)

    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt_v1:
        gt_v1.watch(X_bdy)
        v1 = output(ROT, X_bdy)[:, 0:1]
    v1grad = gt_v1.gradient(v1, X_bdy)

    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt_v2:
        gt_v2.watch(X_bdy)
        v2 = output(ROT, X_bdy)[:, 1:2]
    v2grad = gt_v2.gradient(v2, X_bdy)

    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt_v3:
        gt_v3.watch(X_bdy)
        v3 = output(ROT, X_bdy)[:, 2:3]
    v3grad = gt_v3.gradient(v3, X_bdy)

    vrot = tf.concat([v3grad[:, 1:2] - v2grad[:, 2:3], v1grad[:, 2:3] - v3grad[:, 0:1], v2grad[:, 0:1] - v1grad[:, 1:2]], axis=1)
    sigma = ugrad + vrot

    outernormal = X_bdy

    loss_bdy = bdy_size * tf.reduce_mean(u_bdy * tf.reduce_sum(sigma * outernormal, axis = 1, keepdims = True))

    return loss_pde + loss_bdy

# ------------------- second Adam TRAINING LOOP ---------------------------------------
def trainingrot(GRAD, ROT, X_domain, X_bdy, u_bdy, domain_size, bdy_size, optimizer):
    with tf.GradientTape() as tape:
        loss_value = loss(GRAD, ROT, X_domain, X_bdy, u_bdy, domain_size, bdy_size)
    grads = tape.gradient(loss_value, ROT.trainable_variables)
    optimizer.apply_gradients(zip(grads, ROT.trainable_variables))
    return loss_value

for i in range(Adam_epochs):
    X = generate_inputs(Nint)
    X_bdy, u_bdy = generate_bdy(Nbdy, dim, domain_size, bdy_size)
    r = tf.reduce_sum(Xtest ** 2, axis=1, keepdims=True) ** (1 / 2)

    if (i + 1) % Nprint_adam == 0:
        loss_value = loss(GRAD, ROT, X, X_bdy, u_bdy, domain_size, bdy_size)
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt_u:
            gt_u.watch(Xtest)
            u = output(GRAD, Xtest)
        ugrad = gt_u.gradient(u, Xtest)

        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt_v1:
            gt_v1.watch(Xtest)
            v1 = output(ROT, Xtest)[:, 0:1]
        v1grad = gt_v1.gradient(v1, Xtest)

        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt_v2:
            gt_v2.watch(Xtest)
            v2 = output(ROT, Xtest)[:, 1:2]
        v2grad = gt_v2.gradient(v2, Xtest)

        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt_v3:
            gt_v3.watch(Xtest)
            v3 = output(ROT, Xtest)[:, 2:3]
        v3grad = gt_v3.gradient(v3, Xtest)

        vrot = tf.concat([v3grad[:, 1:2] - v2grad[:, 2:3], v1grad[:, 2:3] - v3grad[:, 0:1], v2grad[:, 0:1] - v1grad[:, 1:2]], axis=1)

        sigma = ugrad + vrot
        sigma = tf.where(r<=1, sigma, 0)
        utest = sigma.numpy()

        error = (np.mean(np.linalg.norm(utest - uexact, axis=1)**q))**(1/q) / (np.mean(np.linalg.norm(uexact, axis=1)**q))**(1/q)
        print("i=", i + 1)
        print(template.format(i + 1, loss_value, error))
        loss_list[i // Nprint_adam] = loss_value.numpy()
        error_list[i // Nprint_adam] = error

    trainingrot(GRAD, ROT, X, X_bdy, u_bdy, domain_size, bdy_size, optimizer_rot)

np.savetxt(f"loss_adam.txt", np.c_[epochs, loss_list, error_list])
with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt_u:
    gt_u.watch(Xtest)
    u = output(GRAD, Xtest)
ugrad = gt_u.gradient(u, Xtest)

with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt_v1:
    gt_v1.watch(Xtest)
    v1 = output(ROT, Xtest)[:, 0:1]
v1grad = gt_v1.gradient(v1, Xtest)

with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt_v2:
    gt_v2.watch(Xtest)
    v2 = output(ROT, Xtest)[:, 1:2]
v2grad = gt_v2.gradient(v2, Xtest)

with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt_v3:
    gt_v3.watch(Xtest)
    v3 = output(ROT, Xtest)[:, 2:3]
v3grad = gt_v3.gradient(v3, Xtest)

vrot = tf.concat([v3grad[:, 1:2] - v2grad[:, 2:3], v1grad[:, 2:3] - v3grad[:, 0:1], v2grad[:, 0:1] - v1grad[:, 1:2]],
                 axis=1)
sigma = ugrad + vrot

output_path = '.'
loss_path = os.path.join(output_path, "loss2.mat")
sio.savemat(loss_path, {"iteration": epochs, "solution_loss": loss_list, "solution_error": error_list})

gradtrue = tf.where(tf.reduce_sum(Xtest**2, axis=1, keepdims = True) < 1, gradsolution(Xtest), 0)
true = tf.where(tf.reduce_sum(Xtest**2, axis=1, keepdims = True) < 1, solution(Xtest), 0)

utest = -tf.reduce_sum(sigma**2, axis=1, keepdims = True)**((q-2)/2) * sigma
utest = tf.where(tf.reduce_sum(Xtest**2, axis=1, keepdims = True) < 1, utest, 0)
utest = utest.numpy()
ugradtest = sigma.numpy()

sio.savemat(os.path.join(output_path, "solution" + ".mat"),
            {"samples": Xtest.numpy(),
             "optimal_grad_solution": gradtrue.numpy(),
             "pred_grad_solution": ugradtest,
             "optimal_solution": true.numpy(),
             "pred_solution": utest})