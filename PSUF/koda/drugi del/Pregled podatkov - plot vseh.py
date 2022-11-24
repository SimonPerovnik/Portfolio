import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
from DataGeneration import theta_time_evolution
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

anchor = 'pp'

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
IntensityNoise = np.load(path + f'\\intensity{lbd}noise100.npy')[0:N]
ThetaZero  = np.load(path + '\\theta0.npy')

# Calculate evoulution of time during simulation

T = dt * num_timesteps
Nt = num_timesteps //nth_step_save
time = np.linspace(0, T, Nt)

# Plot 2000 intensities

part = 1

plt.figure(figsize=(7,3))
for i in Intensity[:2000]:
    plt.plot(time[:Nt//part], i[:Nt//part], alpha=0.03, c='0')
plt.plot(time[:Nt//part],np.mean(Intensity, axis=0)[:Nt//part])
plt.xlabel('$t$ [s]')
plt.ylabel('$I(t)$', rotation=0)
plt.title(f'Sidranje ${lab}$ \n' +'Prvih 2000 $I(t)$ v setu ter povprečje celotnega seta')
plt.tight_layout()
plt.show()

