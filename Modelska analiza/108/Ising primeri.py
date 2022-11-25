import numpy as np
import matplotlib.pyplot as plt
from random import randrange, choice, random
from scipy.stats import chisquare
import seaborn as sn
import os
import csv

#k = 1.38 * 10**(-23)
T = 0.001

N = 50
J = -1
H = 0
steps = 1000000
step_OK = 0

A_0 = [ [choice([1 , -1]) for i in range(N)] for j in range(N)]


def indexValidation(a):
    k, l = a
    try: A[k][l]
    except IndexError:
        return 0
    else:
        return A[k][l]

E_steps = []
E_steps.append(0)
A = np.array(A_0).copy()

fig, (ax1, ax2, ax3)  = plt.subplots(1, 3, sharey='row')
ax1.set_title('število korakov: 0')
ax2.set_title('število korakov: 3x$10^4$')
ax3.set_title('število korakov: 5x$10^4$')
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
for i in range(steps):
    k, l = randrange(0,N), randrange(0,N)
    
    sosedi = [indexValidation(i) for i in[[k-1,l] ,[k+1,l], [k,l-1], [k,l+1]]]
    sosedi = [A[k-1][l] ,A[(k+1)%N][l], A[k][l-1], A[k][(l+1)%N]]
    delta_E = 2*J*A[k][l] * np.sum(sosedi) + 2*H*A[k][l]
    if delta_E <= 0:
        E_steps.append(E_steps[i] + delta_E)
        step_OK += 1
        A[k][l] = -1 * A[k][l]
    elif delta_E > 0:
        cifra = random()
        if  cifra <= np.exp(-1*delta_E/T):
            E_steps.append(E_steps[i] + delta_E)
            A[k][l] = -1 * A[k][l]
            step_OK += 1
        else:
            # Če zavrnemo potezo, obrnemo spin nazaj
            #A[k][l] = -1 * A[k][l]
            E_steps.append(E_steps[i])
    if i == 0:
        ax1.imshow(A, cmap='Greys')
    if i == 3000-1:
        ax2.imshow(A, cmap='Greys')
    if i == steps-1:
        ax3.imshow(A, cmap='Greys_r')
        
fig.suptitle('Isingov model : $J = {}, H = {}, T = {}$\n velikost mreže : ${} x {}$, število korakov: $10^{}$, delež sprejetih: ${}$'.format(J, H, T, N, N, int( np.log10(steps)), step_OK/steps ))
plt.tight_layout()
plt.show()
