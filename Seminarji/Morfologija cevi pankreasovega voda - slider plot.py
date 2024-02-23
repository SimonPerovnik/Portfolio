import numpy as np

def critical_N(alpha):
    return np.pi/(np.arcsin(1/(2*alpha)))

def R_plus(N, alpha):
    phi = np.pi/N
    return ( np.sqrt(np.cos(phi)) * (1 - 2*alpha*np.sin(phi)) ) / (np.sqrt(8*alpha) * np.sin(phi))

def R_minus(N, alpha):
    phi = np.pi/N
    return - ( np.sqrt(np.cos(phi)) * (1 - 2*alpha*np.sin(phi)) ) / (np.sqrt(8*alpha) * np.sin(phi))    

def R(N, alpha):
    phi = np.pi/N    
    
    R = ( np.sqrt(np.cos(phi)) * (1 - 2*alpha*np.sin(phi)) ) / (np.sqrt(8*alpha) * np.sin(phi))
    if R < 0:
        R = - R
        return 1e-12
    return R

def limit_R(N, alpha):
    return N/(np.pi * np.sqrt(8*alpha)) - np.sqrt(alpha/2)

def ration_La_Ll(N, alpha):
    # gledamo razmerje L_a / (L_l / 2), torej razmerje stranic
    phi = np.pi/N
    radius = R(N, alpha)
    return 2*np.sin(phi) / (np.sqrt(1 + 1/(np.tan(phi) * radius**2)) - 1)

def ration_La_Lb(N, alpha):
    # gledamo razmerje L_a / L_b, torej razmerje stranic
    phi = np.pi/N
    radius = R(N, alpha)
    return 1 / np.sqrt(1 + 1/(np.tan(phi) * radius**2))

def polar_to_xy(r, psi_list):
    xy = [[r * np.cos(psi), r * np.sin(psi)] for psi in psi_list]
    return xy

def trapezoids_from_xy(xy_inner, xy_outer):
    N = len(xy_inner)
    trap_list = []
    for i in range(N):
        x = [xy_inner[i][0], xy_inner[(i+1)%N][0], xy_outer[(i+1)%N][0], xy_outer[i][0]]
        y = [xy_inner[i][1], xy_inner[(i+1)%N][1], xy_outer[(i+1)%N][1], xy_outer[i][1]]
        trap_list.append(list(zip(x,y)))
    return trap_list

def trapezoid_coords(N, alpha):
    phi = np.pi/N
    radius = R(N, alpha)
    radius_ = radius/np.cos(phi)
    
    psi_list = [n * 2*np.pi/N for n in range(N)]
    
    r_inner = radius_
    L_l = radius/np.cos(phi) * (np.sqrt(1 + 1/(np.tan(phi) * radius**2)) - 1)
    r_outer = radius_ + L_l
    
    xy_inner = polar_to_xy(r_inner, psi_list)
    xy_outer = polar_to_xy(r_outer, psi_list)

    return trapezoids_from_xy(xy_inner, xy_outer)

def energy(N, alpha):
    phi = np.pi/N
    radius = R(N, alpha)
    
    Ll = radius/np.cos(phi) * (np.sqrt(1 + 1/(np.tan(phi) * radius**2)) - 1)
    LaLb = 2*radius*alpha*np.tan(phi) * (np.sqrt(1 + 1/(np.tan(phi) * radius**2)) + 1)
    
    return Ll + LaLb

def energyR(N, alpha, radius):
    phi = np.pi/N
    
    if radius == 0:
        radius = 0.0001
    Ll = radius/np.cos(phi) * (np.sqrt(1 + 1/(np.tan(phi) * radius**2)) - 1)
    LaLb = 2*radius*alpha*np.tan(phi) * (np.sqrt(1 + 1/(np.tan(phi) * radius**2)) + 1)
    
    return Ll + LaLb    

def energy_derivative(N, alpha, radius):
    
    phi = np.pi/N
    
    if radius == 0:
        radius = 0.0000000001
    
    return (2*alpha*np.sin(phi) + 1) * (np.sqrt(1 + 1/(np.tan(phi) * radius**2)) - 1/np.sqrt(1 + 1/(np.tan(phi) * radius**2)) * (1/(np.tan(phi)*radius**2))) + (2*alpha*np.sin(phi) - 1)


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
import matplotlib

# matplotlib.style.use('seaborn-notebook')

matplotlib.rcParams['savefig.format'] = 'pdf'
matplotlib.rcParams["savefig.directory"] = r"E:\Faks\5.letnik\Biofizika membrac, celic in tkiv\naloga\slike"
# plt.rc('text', usetex=True)
# plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'], 'size': 22})
SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)            # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


fig, ax = plt.subplots(figsize=(7,8.6))

N = 10
alpha = 0.2
radius = R(N, alpha)
coords = trapezoid_coords(N, alpha)
for trapezoid in coords:
    ax.add_patch(patches.Polygon(xy=trapezoid, fill=False, linewidth=3))
minima = np.min([np.transpose(coords)[1], np.transpose(coords)[0]]) - 0.5
maxima = np.max([np.transpose(coords)[1], np.transpose(coords)[0]]) + 0.5
ax.set_ylim(ymin= minima, ymax= maxima)
ax.set_xlim(xmin=minima, xmax=maxima)
ax.set_title('$R_{{\mathrm{{eq}}}} = {}$'.format(round(radius,2)))

plt.subplots_adjust(bottom=0.25)

ax_slider_N = plt.axes([0.1,0.1,0.8,0.05], facecolor='teal')
ax_slider_alpha = fig.add_axes([0.1,0.05,0.8,0.05], facecolor='teal')
slider_alpha = Slider(ax_slider_alpha, r'$\alpha$', valmin=0.1, valmax=1, valstep=0.01, valinit=alpha)    
slider_N = Slider(ax_slider_N, '$N$', valmin=3, valmax=20, valstep=1, valinit=N)


def update_line(val):
    ax.clear()
    coords = trapezoid_coords(slider_N.val, slider_alpha.val)
    radius = R(slider_N.val, slider_alpha.val)
    for trapezoid in coords:
        ax.add_patch(patches.Polygon(xy=trapezoid, fill=False, linewidth=3))
    minima = np.min([np.transpose(coords)[1], np.transpose(coords)[0]]) - 0.5
    maxima = np.max([np.transpose(coords)[1], np.transpose(coords)[0]]) + 0.5
    ax.set_ylim(ymin= minima, ymax= maxima)
    ax.set_xlim(xmin=minima, xmax=maxima)
    ax.set_title('$R_{{\mathrm{{eq}}}} = {}$'.format(round(radius,2)))
    plt.draw()

slider_N.on_changed(update_line)
slider_alpha.on_changed(update_line)
plt.show()