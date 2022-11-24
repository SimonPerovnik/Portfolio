import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
from DataGeneration import theta_time_evolution

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

# Load data

Kvalues = np.load('Podatki\\DataK\\Kvalues.npy')
Intensity = np.load(f'Podatki\\DataK\\intensity{lbd}noise{noise}.npy')
ThetaZero  = np.load('Podatki\\DataK\\theta0.npy')

# Calculate evoulution of time during simulation

T = dt * num_timesteps
Nt = num_timesteps //nth_step_save
time = np.linspace(0, T, Nt)

# Find I(t) that are most different from the mean

I_norm1 = np.sum(np.abs(Intensity - np.mean(Intensity, axis=0)), axis=1)
I_norm2 = np.sum((Intensity - np.mean(Intensity, axis=0))**2, axis=1)

#selection = np.argpartition(I_norm1, -8)[-8:]
selection = np.argpartition(I_norm2, 8)[:8]
print(selection)

# Plot some intensities

part = 1

plt.figure(figsize=(7,3))
for i in selection:
    plt.plot(time[:Nt//part], Intensity[i][:Nt//part], label=f'$K_i={Kvalues[i]/1e-13:.2f}$ pN, i $={i}$', alpha=0.5)
plt.xlabel('$t$ [s]')
plt.ylabel('$I(t)$', rotation=0)
plt.legend(frameon = True, fancybox = True, facecolor='white', framealpha=1, loc='lower left', mode = 'expand', bbox_to_anchor=(0,1.02,1,0.2), ncol=4)
#plt.tight_layout()
plt.show()

# Plot time evolutions of the director field : theta(z,t)

cmap = plt.get_cmap('coolwarm')
plt.grid(False)
for i in range(4):
    theta_t = theta_time_evolution(theta0=ThetaZero[selection[i]], C=Kvalues[selection[i]])
    plt.subplot(2,2,i+1)
    plt.title(f'$K={Kvalues[selection[i]]/1e-13:.2f}$ pN')
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
plt.suptitle(f'Relaksacija direktorskega polja pri $\lambda = {lbd}$nm , noise $={bool(noise)} $')
plt.tight_layout()
plt.show()