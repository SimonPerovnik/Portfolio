import numpy as np
import matplotlib.pyplot as plt
from random import randrange, choice, random
from scipy.stats import chisquare
import seaborn as sn
import os
import csv

k = 1.38 * 10**(-23)
T = 0.01

N = 100
h_max = 18 #Od nič do 18 -> 19 nivojev
alpha = 10
steps = 4000
step_OK = 0
step_NOK = 0

h_0 = [-1 * randrange(0,h_max) for i in range(N)]
h_0[0], h_0[-1] = 0, 0

def energy(h):
    E = 0
    for i in range(1, N-1):
        E += alpha * h[i] + 1/2*(h[i+1] - h[i])**2
    return E


E_steps = []
h = h_0
E_steps.append(energy(h))

for i in range(steps):
    k = randrange(1,N-1)
    delta_k = choice([1 , -1])
    if (h[k] + delta_k) < (-1* h_max) or (h[k] + delta_k) > 0:
        while((h[k] + delta_k) < (-1* h_max) or (h[k] + delta_k) > 0):
            k = randrange(1,N-1)
            delta_k = choice([1 , -1])
    delta_E = delta_k**2 - delta_k* (h[k+1] - 2 *h[k] + h[k-1] - alpha )
    if delta_E <= 0:
        E_steps.append(E_steps[i] + delta_E)
        step_OK += 1
        h[k] += delta_k
    elif delta_E > 0:
        if random() < np.exp(-delta_E/(k*T)):
            E_steps.append(E_steps[i] + delta_E)
            step_OK += 1
            h[k] += delta_k
        else:
            step_NOK += 1
            E_steps.append(E_steps[i])
    if i > 50:
        if np.mean(E_steps[-50:]) - E_steps[-1] < 10:
            SimulFinished = 1
            
plt.plot(range(N), h, marker ='o')
plt.show()

plt.plot(range(steps+1), E_steps )
plt.show()
    