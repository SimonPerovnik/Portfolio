import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
from sklearn.model_selection import GridSearchCV
#from tensorflow.keras.wrappers.scikit_learn import KarasRegressor
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import optimizers
from sklearn.model_selection import KFold
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

matplotlib.style.use('seaborn-darkgrid')
matplotlib.rcParams['savefig.format'] = 'pdf'
matplotlib.rcParams["savefig.directory"] = "E:\\Faks\\5.letnik\\Praktikum strojnega učenja\\3. naloga\\slike"

rootdir = os.getcwd()
os.chdir('E:\\Faks\\5.letnik\\Praktikum strojnega učenja\\3. naloga\\')

PlotData = False
step = -1

# Metadata from generation process

lbd = 500
noise = 0
dt = 4.9999999999999996e-06
num_timesteps = 240000
nth_step_save = 600
z_steps = 200

# Load data

N = int(10000 * 1)
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

n = int(X.shape[0] * 0.8)
X_train, X_test = X[:n], X[n:]
Y_train, Y_test = Y[:n], Y[n:]
 
if PlotData:
    binN = 10
    fig, (ax1,ax2) = plt.subplots(1,2)
    cmap = plt.get_cmap('viridis')
    N_test, bins, patches = ax1.hist(Y_test, bins= np.linspace(0, 1, binN+1), rwidth=0.7, align='mid', density = True)
    for i in range(binN):
        patches[i].set_facecolor(cmap( (N_test[i] - N_test.min()) / (N_test.max() - N_test.min()) ))
    ax1.set_ylabel('delež', rotation=0)
    ax1.set_xlabel('$K/K_{\max}$')
    ax1.set_title('Testni podatki')
    N, bins, patches = ax2.hist(Y_train, bins= np.linspace(0, 1, binN+1), rwidth=0.7, align='mid', density = True)
    for i in range(binN):
        patches[i].set_facecolor(cmap( (N[i] - N.min()) / (N.max() - N.min()) ))
    ax2.set_ylabel('delež', rotation=0)
    ax2.set_xlabel('$K/K_{\max}$')
    ax2.set_title('Učni podatki')
    plt.show()

# Construct NN

def createModel():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(400,)),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dense(400, activation='relu'),
    tf.keras.layers.Dense(1, activation='relu')
    ])

    model.compile(
        optimizer=optimizers.Adam(0.001),
        loss='mean_absolute_error',
        metrics=['mean_squared_error'],
    )

    
    return model

#Implementing cross validation
 
k = 3
kf = KFold(n_splits=k, random_state=None)

scores_per_fold = []
for train , test in kf.split(X_train, Y_train):
    model = createModel()
    #model.summary()
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1,
                                mode='auto', min_delta=0.00001, cooldown=0, min_lr=0)
    
    history = model.fit(X_train[train], Y_train[train],epochs = 100, 
                        batch_size = 32, shuffle = True ,callbacks=[reduce_lr], verbose=0)
    
    scores = model.evaluate(X_train[test], Y_train[test])[1]
    scores_per_fold.append(scores)
    print('Fold completed')

print(np.mean(scores_per_fold))
score = np.mean(scores_per_fold)

model.save('modeli\\model.h5')

Y_predicted = model.predict(X_train[test])
Y_test = Y_train[test]


