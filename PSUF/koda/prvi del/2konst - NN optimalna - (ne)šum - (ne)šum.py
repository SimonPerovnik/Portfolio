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
noise = 100
dt = 4.9999999999999996e-06
num_timesteps = 240000
nth_step_save = 600
z_steps = 200

# Load data

path = 'Podatki\\DataK13_hh'
#path = 'Podatki\\DataK13_ph'
#path = 'Podatki\\DataK13_pp'

N = int(10000 * 1)
Kvalues = np.load(path+'\\Kvalues.npy')[0:N]
Intensity = np.load(path + f'\\intensity{lbd}noise{noise}.npy')[0:N]
IntensityNoise = np.load(path + f'\\intensity{lbd}noise100.npy')[0:N]
ThetaZero  = np.load(path + '\\theta0.npy')

# Calculate evoulution of time during simulation

T = dt * num_timesteps
Nt = num_timesteps //nth_step_save
time = np.linspace(0, T, Nt)

# Prepare data for NN

Kmax = 20e-12

X = Intensity
X_N = IntensityNoise
Y = Kvalues/Kmax

n = int(X.shape[0] * 0.81)
X_train, X_test = X[:n], X[n:]
Y_train, Y_test = Y[:n], Y[n:]
 
# Construct NN

def createModel():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(400,)),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dense(400, activation='relu'),
    tf.keras.layers.Dense(2, activation='relu')
    ])

    model.compile(
        optimizer=optimizers.Adam(0.001),
        loss='mean_absolute_error',
        metrics=['mean_squared_error'],
    )

    
    return model


model = createModel()
#model.summary()
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1,
                            mode='auto', min_delta=0.00001, cooldown=0, min_lr=0)

history = model.fit(X_train, Y_train,epochs = 100, 
                    batch_size = 32, shuffle = True ,callbacks=[reduce_lr], verbose=0)

scores = model.evaluate(X_test, Y_test)[1]
scores = round(scores, 3)
Y_predicted = model.predict(X_test)

# Lasso

class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, alpha_other=0.5):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other
        self.P = 1

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.P =+1
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

def accept(event):
    if event.key == "enter":
        global step
        step = step + 1
        fig.canvas.draw()
        
        Y_pred_selected = selector.xys[selector.ind].T[0]
        Y_pred_ind = []
        for i in Y_pred_selected:
            Y_pred_ind.append(Y_test.tolist().index(i))
        ind_all.append(Y_pred_ind)
        
        # plot intensities
        
        figg, axx = plt.subplots(figsize=(5,2))
        for i in Y_pred_ind:
            plots = axx.plot(time, Intensity[n+i], alpha=0.05, c='0')
        axx.plot(time,np.mean(Intensity[n + np.array(Y_pred_ind)], axis=0), c='C' +str(step))
        axx.set_xlabel('$t$ [s]')
        axx.set_ylabel('$I(t)$', rotation=0)
        #axx.set_title('$I(t)$ za izbrane $K$ iz podmnožice napačno napovedanih')
        plt.tight_layout()
        plt.show()
        
fig, ax = plt.subplots()
ax.set_aspect('equal')
pts = ax.scatter(Y_test, Y_predicted, s=0.3)
ax.set_ylabel('$K_{predicted}$')
ax.set_xlabel('$K_{real}$')
ax.set_title(f'Primerjava napovedanih in pravih vrednosti (MSE = ${scores}$) \n train: Šum - test: Šum')

selector = SelectFromCollection(ax, pts)
ind_all = []

fig.canvas.mpl_connect("key_press_event", accept)
plt.show()

plt.scatter(Y_test, Y_predicted, s=0.3)
j = 0
for i in ind_all:
    plt.scatter(Y_test[i], Y_predicted[i], s=3, c = 'C' + str(j))
    j += 1
plt.ylabel('$K_{predicted}$')
plt.xlabel('$K_{real}$')
plt.title(f'Primerjava napovedanih in pravih vrednosti (MSE = ${scores}$) \n train: Šum - test: Šum')
plt.gca().set_aspect('equal')
plt.show()



