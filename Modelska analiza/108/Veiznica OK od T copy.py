import numpy as np
import matplotlib.pyplot as plt
from random import randrange, choice, random
from scipy.stats import chisquare
import seaborn as sn
import os
import csv

k = 1.38 * 10**(-23)
T = 0.01

N = 17
h_max = 18 #Od nič do 18 -> 19 nivojev
alpha = 1
steps = 1000
step_OK = 0
step_NOK = 0

def energy(h):
    E = 0
    for i in range(1, N-1):
        E += alpha * h[i] + 1/2*(h[i+1] - h[i])**2
    return E
T_list = np.logspace(-2, 2, 50)
alpha_list = [0.1,1,3,5,10]
for alpha in alpha_list:
    E_all = []
    E_all_mean = []
    step_OK_all = []
    for T in T_list:
        E_mean = []
        h_0 = [-1 * randrange(0,h_max) for i in range(N)]
        h_0[0], h_0[-1] = 0, 0
        E_steps = []
        h = h_0
        E_steps.append(energy(h))
        SimulFinished = 0
        step_OK = 0

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
                E_mean.append(np.mean(E_steps[-300:]))
                if np.mean(E_steps[-50:]) - E_steps[-1] < 10:
                    SimulFinished = 1
                    
        step_OK_all.append(step_OK/steps)
        
        # plt.plot(range(N), h, marker='o', label= 'T = ' + str(T), c=cmap(T_list.index(T)/len(T_list)))
        E_all_mean.append(E_mean)
        E_all.append(E_steps)
    cmap = plt.get_cmap('viridis')
    plt.plot(T_list, step_OK_all, label='$\\alpha$ = ' + str(alpha), c=cmap(alpha_list.index(alpha)/len(alpha_list)), marker= 'o', linewidth=0.9, markersize=4)

plt.title('Molekulska verižnica - število sprejetih korakov \n' + '$h_{max}' +' = ${}, število korakov: {}'.format(h_max, steps))
plt.legend(frameon = True, fancybox = True, facecolor='white', framealpha=1)
plt.ylabel('$korak_{OK}(T)$', rotation=0)
plt.xscale('log')
plt.xlabel('$T$')
plt.show()




    # plt.title('Molekulska verižnica - končno stanje\n' + '$h_{max}' +' = ${}, $\\alpha = ${}'.format(h_max, alpha))
    # plt.legend(frameon = True, fancybox = True, facecolor='white', framealpha=1)
    # plt.xlabel('$i$')
    # plt.ylabel('$h_i$', rotation=0)
    # plt.show()
    # for i in E_all:
    #     plt.plot(range(steps+1), i, label= 'T = ' + str(T_list[E_all.index(i)]), c=cmap(E_all.index(i)/len(E_all)))
    # plt.legend(frameon = True, fancybox = True, facecolor='white', framealpha=1)
    # plt.xlabel('korak')
    # plt.xscale('log')
    # plt.ylabel('$E$', rotation=0)
    # plt.show()

    # for i in E_all_mean:
    #     plt.plot(range(len(i)), i, label= 'T = ' + str(T_list[E_all_mean.index(i)]), c=cmap(E_all_mean.index(i)/len(E_all_mean)))
    # plt.legend(frameon = True, fancybox = True, facecolor='white', framealpha=1)
    # plt.xlabel('korak')
    # plt.xscale('log')
    # plt.ylabel('$E$', rotation=0)
    # plt.show()

    