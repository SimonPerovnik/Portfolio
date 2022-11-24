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

N = int(10000 * 0.3)
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
    tf.keras.layers.Dense(200, activation=actF),
    tf.keras.layers.Dense(100, activation=actF),
    tf.keras.layers.Dense(1, activation=actF)
    ])

    if optimizer == 'adam':
        optimizerM = tf.keras.optimizers.Adam(learning_rate=eta)
    if optimizer == 'adagrad':
        optimizerM = tf.keras.optimizers.Adagrad(learning_rate=eta)
    if optimizer == 'SGD':
        optimizerM = tf.keras.optimizers.SGD(learning_rate=eta)
    model.compile(
        optimizer=optimizerM,
        loss='mean_absolute_error',
        metrics=['mean_squared_error'],
    )

    
    return model

eta = 0.001
actF='relu'
opt_list = ['adam', 'adagrad', 'SGD']
ep_list = [5,15,30,50,100]
for optimizer in opt_list:
  acc_list = []
  acc_train_list = []
  for epoh in ep_list:


    #Implementing cross validation
    
    k = 3
    kf = KFold(n_splits=k, random_state=None)

    scores_per_fold = []
    for train , test in kf.split(X, Y):
        model = createModel()
        #model.summary()
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1,
                                    mode='auto', min_delta=0.00001, cooldown=0, min_lr=0)
        
        history = model.fit(X[train], Y[train],epochs = epoh, 
                            batch_size = 32, shuffle = True ,callbacks=[reduce_lr],
                            verbose=0, )
        
        scores = model.evaluate(X[test], Y[test])[1]
        scores_per_fold.append(scores)
        #print('Fold completed')
    fin_score = np.mean(scores_per_fold)
    print(epoh)
    acc_list.append(fin_score)
    acc_train_list.append(history.history['mean_squared_error'][-1])
    
  plt.plot(ep_list, acc_list,label=optimizer + ' - train', marker='o')
  plt.plot(ep_list, acc_train_list,label=optimizer + ' - test', ls='dotted', color= 'C' + str(opt_list.index(optimizer)))
  print('*************************************************************************************************************')
  print(optimizer)

plt.legend(frameon = True, fancybox = True, facecolor='white', framealpha=1)
plt.title('NN za različno število epoh$')
#plt.xscale('log')
plt.xlabel('hitrost učenja')
plt.ylabel('MSE')
plt.show()

