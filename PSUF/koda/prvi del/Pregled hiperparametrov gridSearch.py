import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
from sklearn.model_selection import GridSearchCV
#from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from scikeras.wrappers import KerasClassifier, KerasRegressor
import scikeras
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import optimizers
from sklearn.model_selection import KFold
from sklearn.metrics import SCORERS

import warnings
warnings.filterwarnings(action='ignore')


matplotlib.style.use('seaborn-darkgrid')
matplotlib.rcParams['savefig.format'] = 'pdf'
matplotlib.rcParams["savefig.directory"] = "E:\\Faks\\5.letnik\\Praktikum strojnega učenja\\3. naloga\\slike"

rootdir = os.getcwd()
os.chdir('E:\\Faks\\5.letnik\\Praktikum strojnega učenja\\3. naloga\\')

PlotData = False

# Metadata from generation process

lbd = 500
noise = 0
dt = 4.9999999999999996e-06
num_timesteps = 240000
nth_step_save = 600
z_steps = 200

# Load data

N = int(10000 * 0.05)
Kvalues = np.load('Podatki\\DataK\\Kvalues.npy')[0:N]
Intensity = np.load(f'Podatki\\DataK\\intensity{lbd}noise{noise}.npy')[0:N]
ThetaZero  = np.load('Podatki\\DataK\\theta0.npy')

# Calculate evoulution of time during simulation

T = dt * num_timesteps
Nt = num_timesteps //nth_step_save
time = np.linspace(0, T, Nt)

# Prepare data for NN

Kmax = 20e-12

X = Intensity
Y = Kvalues/Kmax

n = int(X.shape[0] * 0.9)
X_train, X_test = X[:n], X[n:]
Y_train, Y_test = Y[:n], Y[n:]

# Construct NN

def createModel(activation='relu', optimizer = 'adam', num1=200, num2=400, depth=0, learn_rate=0.001):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(400,)))
    model.add(tf.keras.layers.Dense(num1, activation=activation))
    for i in range(depth):
        model.add(tf.keras.layers.Dense(num1, activation=activation))
    model.add(tf.keras.layers.Dense(num2, activation=activation))
    model.add(tf.keras.layers.Dense(1, activation=activation))

    model.compile(
        optimizer=optimizer,
        loss='mean_absolute_error',
        metrics=['mean_squared_error'],
    )      
    return model

# Implementing grid search

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1,
                            mode='auto', min_delta=0.00001, cooldown=0, min_lr=0)

model = KerasRegressor(model=createModel,epochs=20, batch_size = 32,
                         verbose=0, callbacks=[reduce_lr])

# define the grid search parameters
activation = ['relu', 'tanh', 'sigmoid']
optimizer = ['SGD', 'Adagrad', 'Adam']
learn_rate = [0.001, 0.01, 0.1]
batch_size = [16,32,50]
epochs = [40]
num1 = [100,200,400]
num2 = [100,200,400]
depth = [0,1,2,5]
param_grid = dict(model__activation=activation, model__optimizer=optimizer, batch_size=batch_size, model__num1=num1, model__num2=num2)
param_grid = dict(optimizer__learn_rate = learn_rate, model__num1=num1, model__num2=num2, model__depth = depth)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


