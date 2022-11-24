import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
from DataGeneration import theta_time_evolution2
import numba
from numba import njit
import time
from joblib import Parallel, delayed
from typing import Union

matplotlib.style.use('seaborn-darkgrid')
matplotlib.rcParams['savefig.format'] = 'pdf'
matplotlib.rcParams["savefig.directory"] = "E:\\Faks\\5.letnik\\Praktikum strojnega učenja\\3. naloga\\slike"

rootdir = os.getcwd()
os.chdir('E:\\Faks\\5.letnik\\Praktikum strojnega učenja\\3. naloga\\')

# Metadata from generation process

lbd = 500
noise = 0
dt = 4.9999999999999996e-06
num_timesteps = 240000
nth_step_save = 600
z_steps = 200


anchor_list = ['pp', 'ph', 'hh', 'ph']
N = int(10000 * 1)

def loadData(i):
    anchor = anchor_list[i]
    # Load data
    if anchor == 'hh':
        path = 'Podatki\\DataK13_hh'
        lab = 'H-H'
    if anchor == 'ph':   
        path = 'Podatki\\DataK13_ph'
        lab = 'P - H'
    if anchor == 'pp':
        path = 'Podatki\\DataK13_pp'
        lab = 'P - P'

    N = int(10000 * 1)
    Kvalues = np.load(path+'\\Kvalues.npy')[0:N]
    Intensity = np.load(path + f'\\intensity{lbd}noise{noise}.npy')[0:N]
    ThetaZero  = np.load(path + '\\theta0.npy')
    return anchor, Kvalues, Intensity, ThetaZero

# Calculate evoulution of time during simulation

T = dt * num_timesteps
Nt = num_timesteps //nth_step_save
time = np.linspace(0, T, Nt)

selection = [150, 300, 43, 412, 555]    # Some hand-picked interesting cases

# Plot time evolutions of the director field : theta(z,t)

cmap = plt.get_cmap('coolwarm')
plt.grid(False)
for i in range(4):

    anchor, Kvalues, Intensity, ThetaZero = loadData(i)
    
    label = f'$K^{{{anchor}}}_{{{11}}}$'+f'$={Kvalues[i][0]/1e-13:.2f}$ pN, ' + f'$K^{{{anchor}}}_{{{33}}}$'+f'=${Kvalues[i][1]/1e-13:.2f}$ pN'
    theta_t = theta_time_evolution2(theta0=ThetaZero[selection[i]], C=Kvalues[selection[i]])
    plt.subplot(2,2,i+1)
    plt.title(label)
    plt.xlabel('$z[ \\mu$m]')
    plt.ylabel('$\\theta (z)$', rotation=0)
    # Normalizer
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.2)
    # creating ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ticks=np.linspace(0, 1.2, 4))
    cb.set_label('$t$', rotation=0)
    j = 0
    for theta in theta_t:
        plt.plot(np.linspace(0, 10, z_steps), theta, c = cmap(j / len(theta_t)) )
        j += 1
plt.suptitle(f'Različna sidranja\n' + f'Relaksacija direktorskega polja pri $\lambda = {lbd}$nm')
plt.tight_layout()
plt.show()
