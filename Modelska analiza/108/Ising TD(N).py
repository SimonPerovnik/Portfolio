import numpy as np
import matplotlib.pyplot as plt
from random import randrange, choice, random
from scipy.stats import chisquare
import seaborn as sn
import os
import csv
from matplotlib import cycler
colors = cycler('color',
                ['#EE6666', '#3388BB', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
       axisbelow=True, grid=True, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='gray')
plt.rc('ytick', direction='out', color='gray')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)
#k = 1.38 * 10**(-23)

J = 1
H = 0
steps = 100000
step_OK = 0



def magnetisation(A):
    return np.abs(1/(N)*np.sum(A))

def chi(M_mean_list, M_mean_sq_list, T):
    return (np.mean(M_mean_sq_list) - np.mean(M_mean_list)**2)/(N*T)

def heatCapacity(E_mean_list, E_mean_sq_list, T):
    return (np.mean(E_mean_sq_list) - np.mean(E_mean_list)**2)/(N*T*T)

def indexValidation(a):
    k, l = a
    try: A[k][l]
    except IndexError:
        return 0
    else:
        return A[k][l]

figure, axis = plt.subplots(2, 2)

T_list = np.linspace(1, 5, 15).tolist()
for N in [4,6,10]:
    
    magnetisation_list = []
    chi_list = []
    heatCap_list = []
    E_list = []
    for T in T_list:
        A_0 = [ [choice([1 , -1]) for i in range(N)] for j in range(N)]
        E_steps = []
        E_steps.append(0)
        A = np.array(A_0).copy()
        E_mean_list = []
        M_mean_list = []
        E_mean_sq_list = []
        M_mean_sq_list = []
        for i in range(steps):
            k, l = randrange(0,N), randrange(0,N)
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
            if i > 90000 and i%100 == 0:
                E_mean_list.append(E_steps[i])
                E_mean_sq_list.append(E_steps[i]**2)
                M_mean_list.append(magnetisation(A))
                M_mean_sq_list.append(magnetisation(A)**2)
        magnetisation_list.append(np.mean(M_mean_list)/N)
        chi_list.append(chi(M_mean_list, M_mean_sq_list, T))
        heatCap_list.append(heatCapacity(E_mean_list, E_mean_sq_list, T))
        E_list.append(np.mean(E_mean_list))
    
    plt.style.use('seaborn')  
    axis[0,0].plot(T_list, magnetisation_list, label='$N = {}$'.format(N), marker='o')
    axis[0,0].set_title('$\langle M \\rangle (T)$')
    
    axis[0,1].plot(T_list, chi_list, label='$N = {}$'.format(N), marker='o')
    axis[0,1].set_title('$\langle \chi \\rangle (T)$')
    
    axis[1,0].plot(T_list, heatCap_list, label='$N = {}$'.format(N), marker='o')
    axis[1,0].set_title('$\langle C \\rangle (T)$')
    
    axis[1,1].plot(T_list, E_list, label='$N = {}$'.format(N), marker='o')
    axis[1,1].set_title('$\langle E \\rangle (T)$')
    
    print(N)
axis[0,0].axvline(2.2692, label='$T_C$', ls = ':', c = 'plum')
axis[0,1].axvline(2.2692, label='$T_C$', ls = ':', c = 'plum')
axis[1,0].axvline(2.2692, label='$T_C$', ls = ':', c = 'plum')
axis[1,1].axvline(2.2692, label='$T_C$', ls = ':', c = 'plum')    
axis[0,0].legend(loc='best', frameon = True, fancybox = True, facecolor='white', framealpha=1)
figure.suptitle('Isingov model : $J = {}, H = {}$\nštevilo korakov: ${} \cdot 10^{}$'.format(J, H, int(str(steps)[0]),int( np.log10(steps))))
plt.show()