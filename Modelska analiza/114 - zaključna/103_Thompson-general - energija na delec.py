import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import numpy as np
import math as m
import itertools
from scipy.spatial import ConvexHull, Delaunay

matplotlib.style.use('seaborn')
matplotlib.rcParams['savefig.format'] = 'pdf'
matplotlib.rcParams["savefig.directory"] = r"E:\Faks\4.letnik\Modelska_analiza_I\114 - zaključna\slike"

PlotAnything = 0
Plotting = 0
PlottingVolume = 0
PlottingPhase = 0
N = 15 #Number of charged particles
A = 1 #Elipsoid dimensions
B = 1
C = 1

method_of_choice = 'Nelder-Mead'
#method_of_choice = 'SLSQP'
#method_of_choice = 'Powell'


def s(x):
    return m.sin(x)

def c(x):
    return m.cos(x)

def distanceFactor (x):
    return m.sqrt(  (A*(c(x[0]) * s(x[1])) - A*(c(x[2]) * s(x[3])) )**2 
                +   (B*(s(x[0]) * s(x[1])) - B*(s(x[2]) * s(x[3])) )**2
                +   (C*(c(x[1]) - c(x[3])))**2
                )**(-1)

def callbackF(X):
    data.append(X)
    return None

def elEnergy(*arg):
    sum = 0
    coord = arg
    coords = [[0,0] for i in range(N)]
    for i in range(0,2*N ,2):
        k = i//2
        coords[k][0] = coord[0][i]
        coords[k][1] = coord[0][i+1]
    combs = [i for i in itertools.combinations(coords, 2)]
    combs_flat = np.array(combs).flatten()

    for i in range(0,len(combs_flat),4):
        r = distanceFactor( [combs_flat[i], combs_flat[i+1],combs_flat[i+2],combs_flat[i+3]] )
        #phi = np.abs(combs_flat[i+1] - combs_flat[i+3])
        
        x1, y1, z1 = polarToCartesian(combs_flat[i], combs_flat[i+1])
        x2, y2, z2 = polarToCartesian(combs_flat[i+2], combs_flat[i+3])
        phi = np.arccos(x1*x2 + y1*y2 + z1*z2)
        
        sum += r *c(phi)**2
        #sum += r
    return sum

def elEnergyOld(*arg):
    sum = 0
    coord = arg
    coords = [[0,0] for i in range(N)]
    for i in range(0,2*N ,2):
        k = i//2
        coords[k][0] = coord[0][i]
        coords[k][1] = coord[0][i+1]
    combs = [i for i in itertools.combinations(coords, 2)]
    combs_flat = np.array(combs).flatten()

    for i in range(0,len(combs_flat),4):
        r = distanceFactor( [combs_flat[i], combs_flat[i+1],combs_flat[i+2],combs_flat[i+3]] )
        #phi = np.abs(combs_flat[i+1] - combs_flat[i+3])
        
        x1, y1, z1 = polarToCartesian(combs_flat[i], combs_flat[i+1])
        x2, y2, z2 = polarToCartesian(combs_flat[i+2], combs_flat[i+3])
        phi = np.arccos(x1*x2 + y1*y2 + z1*z2)
        
        #sum += r *c(phi)**2
        sum += r
    return sum

def rotatePoints(points):
    rotatedPoints = []
    for i in range (0, len(points), 2):
        rotatedPoints.append(points[i]+ points[0]*(-1))
        rotatedPoints.append(points[i+1] + points[1]* (-1))
    return rotatedPoints

def polarToCartesian(x1, x2):
    x = A*c(x1) * s(x2)
    y = B*s(x1) * s(x2)
    z = C*c(x2)
    return [x,y,z] 


N_list = range(16,25)
D_list = []
energy_list = []
D_listO = []
energy_listO = []

First = 1

for N in N_list:

    if First:
        #Random starting points
        start = [0,0] #prvi delec fiksiramo na 0,0
        for i in range(N-1):
            start.append(np.random.uniform(0,2*m.pi))
            #start.append((2*m.pi)*(i+1)/N)
            start.append(np.random.uniform(0,m.pi))
        First = 0
    else:
        start = rotatedResult 
        start.append(np.random.uniform(0,2*m.pi))
        start.append(np.random.uniform(0,m.pi))
        
    #Data points memory
    data = []
    NumPaths = N*(N-1)/2
    result = minimize(elEnergy, start, method= method_of_choice, options={'fatol': 0.00001, 'adaptive': True, 'disp': False}, callback=callbackF)
    rotatedResult = rotatePoints(result.x)
    R = elEnergyOld(rotatedResult)
    energy = elEnergy(rotatedResult)
    energy_list.append(energy/N)
    D_list.append(1/R*(NumPaths))

    result = minimize(elEnergyOld, start, method= method_of_choice, options={'fatol': 0.00001, 'adaptive': True, 'disp': False}, callback=callbackF)
    rotatedResult = rotatePoints(result.x)

    energy = elEnergyOld(rotatedResult)
    energy_listO.append(energy/N)
    D_listO.append(1/energy*(NumPaths))
    
    print(N)

plt.plot(N_list, D_list, label='nov model', marker= 'o')
plt.plot(N_list, D_listO, label='star model', marker= 'o')
plt.legend(frameon = True, fancybox = True, facecolor='white', framealpha=1)
plt.xlabel('$N$')
plt.ylabel('$\langle d \\rangle$', rotation=0)
plt.title('Povprečna parska razdalja - primerjava med modeloma')
plt.show()

plt.plot(N_list, energy_list, label='nov model', marker= 'o')
plt.plot(N_list, energy_listO, label='star model', marker= 'o')
plt.legend(frameon = True, fancybox = True, facecolor='white', framealpha=1)
plt.xlabel('$N$')
plt.ylabel('$E/N$', rotation=0)
plt.title('Minimalna energija - primerjava med modeloma')
plt.show()

if PlotAnything:
    
    #Preparing data for plotting
    xx,yy,zz = [],[],[]
    for i in range(0,len(rotatedResult),2):
        if (A!=1 or B!=1 or C!=1):  #na elipsi ne fiksiramo delca na 0,0
            rotatedResult = result.x
        xx.append(polarToCartesian(rotatedResult[i], rotatedResult[i+1])[0])
        yy.append(polarToCartesian(rotatedResult[i], rotatedResult[i+1])[1])
        zz.append(polarToCartesian(rotatedResult[i], rotatedResult[i+1])[2])

    #Preparing data for convex hull function
    points = np.zeros((N,3))
    for i in range(N):
        points[i][0] = xx[i]
        points[i][1] = yy[i]
        points[i][2] = zz[i]


    #Plotting a sphere
    fig = plt.figure(figsize=plt.figaspect(1)*2)
    ax = fig.add_subplot(projection='3d')
    r = 0.05
    u, v = np.mgrid[0:2 * np.pi:60j, 0:np.pi:40j]
    x3 = A*np.cos(u) * np.sin(v)
    y3 = B*np.sin(u) * np.sin(v)
    z3 = C*np.cos(v)
    ax.plot_surface(x3, y3, z3, alpha = 0.3, cmap = 'twilight')
    ax.view_init(90, 0)

    #Plotting particles
    ax.set_box_aspect([1,1,1])
    ax.scatter(xx,yy,zz, s=100)

    if N>3:
        #Constructing and plotting convex hull
        hull = ConvexHull(points)
        for s in hull.simplices:
            s = np.append(s, s[0])  # Here we cycle back to the first coordinate
            ax.plot(points[s, 0], points[s, 1], points[s, 2], "r-")


    if Plotting:
        #Configuring plot
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_zlim([-1,1])
        ax.grid(False)
        plt.axis('off')

        #Showing plot
        plt.show()

    matplotlib.style.use('seaborn')
    if PlottingPhase:
        t = np.arange(len(data))
        data = np.transpose(data)
        for i in range(0,2*N,2):
            plt.scatter(data[i], data[i+1],s=10, c=t, cmap='viridis')
        plt.show()


    #Calculating  hull volume 
    def tetrahedron_volume(a, b, c, d):
        return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6

    def convex_hull_volume(pts):
        ch = ConvexHull(pts)
        dt = Delaunay(pts[ch.vertices])
        tets = dt.points[dt.simplices]
        return np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1],
                                        tets[:, 2], tets[:, 3]))

    if PlottingVolume:
        N_available = [4,5,6,7,8,9,10,12,20,30,50,70,100]
        a_mean = []
        for i in N_available:
            a = []
            f = open("N" + str(i) + "volumen.txt", "r")
            for line in f:
            #f = open("\\Spin\\PerovnikS17$\\_System\\Desktop\\Modelska analiza\\103\\N4volumen.txt", "r")
                value = line.strip()
                a.append(float(value))
            a_mean.append(np.mean(a)/4.1887)
        plt.clf()
        plt.plot(N_available, a_mean, label = 'V(N)/V_UnitSphere')
        plt.xlabel('N')
        plt.ylabel('Prostornina')
        plt.title('Prostornina s točkami zaobjetega prostora sfere')
        plt.plot([0,100],[1, 1], ls = '--')
        plt.xscale('log')
        plt.legend()
        plt.show()
        

