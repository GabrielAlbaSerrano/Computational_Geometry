"""
Practica 4 - GCOM: Transformacion isometrica afin
NOMBRE: Gabriel Alba Serrano
SUBGRUPO: U1
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation
from skimage import io, color
from scipy.spatial import distance

###   APARTADO 1   ###

#Parametrizacion de un cono
def param_cone(u,v):
    u,v = np.meshgrid(u,v)
    x = v*np.cos(u)
    y = v*np.sin(u)
    z = v
    return x.reshape(-1), y.reshape(-1), z.reshape(-1)

#Transformacion isometrica afin de un sistema de 3 dimensiones que consiste en
# una rotación entorno al centroide compuesta con una traslación del vector v
def transf_isom(x,y,z,M,C,v):
    N = len(x)
    xt = x*0
    yt = y*0
    zt = z*0
    for i in range(N):
        q = np.array([x[i],y[i],z[i]])
        xt[i], yt[i], zt[i] = C + np.matmul(M, (q-C)) + v
    return xt, yt, zt

#Grafica de la transformacion del cono para cada tiempo t 
def animate1(t):
    x,y,z = param_cone(np.linspace(0, 2*np.pi, 20),np.linspace(0, 5, 20))
    theta = 3*np.pi
    #Matriz de rotacion de angulo theta en sentido antihorario segun t
    Mt = np.array([[np.cos(t*theta), -np.sin(t*theta), 0], 
                    [np.sin(t*theta), np.cos(t*theta),0],[0,0,1]])
    N = len(x)
    #Centroide del cono
    C = np.array([sum(x)/N, sum(y)/N, sum(z)/N])
    #Diametro mayor del cono, calculado aplicando el Teorema de Pitagoras,
    #ya que el valor maximo de las coordenadas x,y,z es igual a 5 para cada
    #una de ellas, entonces al cortar el cono por el eje Z, obtenemos un
    #triangulo rectangulo en el plano XZ cuya hipotenusa es el diametro mayor d
    d = np.sqrt(2)*5
    vt = np.array([0,0,d])*t
    xt, yt, zt = transf_isom(x,y,z,Mt,C,vt)
    
    ax = plt.axes(xlim=(-12,12), ylim=(-12,12), zlim=(0,12), projection='3d')
    #ax.view_init(60, 30)
    ax.plot_trisurf(xt,yt,zt)
    return ax,


def init1():
    return animate1(0),

"""
cono_t0 = plt.figure()
animate1(0)
plt.title('Transformación isométrica afín del cono para t=0')
cono_t0.savefig('cono_t=0.jpg')

cono_t1 = plt.figure()
animate1(1)
plt.title('Transformación isométrica afín del cono para t=1')
cono_t1.savefig('cono_t=1.jpg')


fig = plt.figure(figsize=(8,8))
ani1 = animation.FuncAnimation(fig, animate1, frames=np.arange(0,1,0.025), 
                               init_func=init1, interval=20)
ani1.save("p4i.gif", fps = 10)
"""

###   APARTADO 2   ###

#Obtenemos las coordenadas de la imagen 'arbol.png'
img = io.imread('arbol.png')
fig = plt.figure()
xyz = img.shape
x = np.arange(0,xyz[0],1) #coordenadas x de cada pixel
y = np.arange(0,xyz[1],1) #coordenadas y de cada pixel
xx,yy = np.meshgrid(x, y)
xx = np.asarray(xx).reshape(-1)
yy = np.asarray(yy).reshape(-1)
z  = img[:,:,1] #componente de color verde de cada pixel
zz = np.asarray(z).reshape(-1)

#Obtenemos las coordenadas del subsistema sigma, formado por los píxeles con
#el color verde menor que 240
x0 = xx[zz<240]
y0 = yy[zz<240]
z0 = zz[zz<240]/256.

#Calculamos el diámetro mayor de los pixeles del sistema sigma
l=zip(x0,y0)
dist=[]
for p in l:
    for q in l:
        dist.append(distance.euclidean(p,q))
d = max(dist)

#Grafica de la transformacion del arbol para cada tiempo t 
def animate2(t):
    theta = 3*np.pi
    #Matriz de rotación de ángulo theta en sentido antihorario según t
    Mt = np.array([[np.cos(t*theta), -np.sin(t*theta), 0], 
                    [np.sin(t*theta), np.cos(t*theta),0],[0,0,1]])
    N = len(x0)
    #Centroide del arbol: la componente Z del centroide es 0, porque la imagen
    #'arbol.png' es un sistema en 2 dimensiones, que no tiene altura
    C = np.array([sum(x0)/N, sum(y0)/N, 0])
    v = np.array([d,d,0])*t
    
    ax = plt.axes(xlim=(0,750), ylim=(0,400), projection='3d')
    #ax.view_init(60, 30)

    XYZ = transf_isom(x0, y0, z0, Mt, C, v)
    col = plt.get_cmap("viridis")(np.array(0.1+z0))
    ax.scatter(XYZ[0],XYZ[1],c=col,s=0.1,animated=True)
    return ax,

def init2():
    return animate2(0),

"""
arbol_t0 = plt.figure()
animate2(0)
plt.title('Transformación isométrica afín del árbol para t=0')
arbol_t0.savefig('arbol_t=0.jpg')

arbol_t1 = plt.figure()
animate2(1)
plt.title('Transformación isométrica afín del árbol para t=1')
arbol_t1.savefig('arbol_t=1.jpg')


fig = plt.figure(figsize=(8,8))
ani2 = animation.FuncAnimation(fig, animate2, frames=np.arange(0,1,0.025), 
                               init_func=init2, interval=20)

ani2.save("p4ii.gif", fps = 10)  
"""