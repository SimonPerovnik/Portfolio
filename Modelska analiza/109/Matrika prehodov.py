import random as rnd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from scipy.sparse import diags

matplotlib.style.use('seaborn')
PlotM = 1
PlotDt = 0

beta = 1
beta_r = 4*beta
beta_s = 5*beta
N = 5
N_all = [25,50,100,150,200]
interval = N*3
N_ext = N + interval
dt_all= [0.0001,0.0005,0.001,0.1]
accuracy = 0.999

def TransitionMatrix (N_ext):
    N = N_ext
    diag1 = np.array([beta_r * n * dt for n in range(0, N)])
    diag2 = np.array([1 - (beta_r + beta_s)* n * dt for n in range(0, N+1)])
    diag3 = np.array([beta_s * n * dt for n in range(1, N+1)])
    diag3[-1] = N*5*dt
    diag2[-1] = 1-N*5*dt


    k = [diag1,diag2,diag3]
    offset = [-1,0,1]
    return diags(k,offset).toarray()

if PlotDt:

    for dt in dt_all[::-1]:
        t_all = []
        for N in N_all:
            interval = N*3
            N_ext = N + interval

            M = TransitionMatrix(N_ext)
            x = np.concatenate((np.zeros(N), np.ones(1), np.zeros(interval)), axis=None)

            t_now = 0
            while (x[0] < accuracy):
                if t_now > 15: break 
                t_now = t_now + dt
                x = M.dot(x)
            print(N)
            t_all.append(t_now)
        plt.scatter(N_all, t_all, label = '$dt = $' + str(dt))
        
        
    plt.xlabel('$N$')
    plt.ylabel('$t_{death}$')
    plt.title('Čas izumrtja glede na velikost populacije pri verjetnosti ' + str(accuracy*100)+'%')
    plt.legend(frameon = True, fancybox = True, facecolor='white', framealpha=1)
    plt.show()

if PlotM:
    fig, (ax1, ax2, ax3)  = plt.subplots(1, 3, sharey='row')
    ax1.set_title('$dt = 0.0001$')
    ax2.set_title('$dt = 0.001$')
    ax3.set_title('$dt = 0.01$')
    dt=0.0001
    ax1.axis('off')
    M = TransitionMatrix(N_ext)
    c = ax1.imshow(M, cmap='plasma', vmin=0, vmax=1)
    dt=0.01
    ax2.axis('off')
    M = TransitionMatrix(N_ext)
    d = ax2.imshow(M, cmap='plasma', vmin=0, vmax=1)
    dt=0.1
    ax3.axis('off')
    M = TransitionMatrix(N_ext)
    e = ax3.imshow(M, cmap='plasma', vmin=0, vmax=1)
    fig.colorbar(d, orientation = 'horizontal', ax=(ax1, ax2, ax3))
    fig.suptitle('Matrike prehodov za $\u03B2_r =$ ' + str(beta_r) + ', $\u03B2_s =$ ' + str(beta_s))
    

    plt.show()

