import numpy as np
import matplotlib.pyplot as plt
from random import randrange, choice, random
from scipy.stats import chisquare
import seaborn as sn
import os
import csv

#k = 1.38 * 10**(-23)
T = 0.1

N = 50
J = 1
H = 0
steps = 500000
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

for i in range(steps):
    k, l = randrange(0,N), randrange(0,N)
    
    sosedi = [indexValidation(i) for i in[[k-1,l] ,[k+1,l], [k,l-1], [k,l+1]]]
    #sosedi = [A[k-1][l] ,A[(k+1)%N][l], A[k][l-1], A[k][(l+1)%N]]
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

plt.title('Isingov model - $J = {}, H = {}, T = {}$\n velikost mreže : ${} x {}$, število korakov: $10^{}$'.format(J, H, T, N, N, int( np.log10(steps)) ))
sn.heatmap(A, annot=False, fmt='g', cbar=False, square=True)
plt.tight_layout()
plt.xticks(ticks=[])
plt.yticks(ticks=[])
plt.show()
print(step_OK/steps)
plt.plot(range(steps), E_steps)
#plt.xscale('log')
plt.show()