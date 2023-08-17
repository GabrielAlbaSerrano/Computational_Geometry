"""
GCOM - PRACTICA 3: Discretización de sistemas continuos y Teorema de Liouville
NOMBRE: Gabriel Alba Serrano
SUBGRUPO: U1
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from matplotlib import animation

# En primer lugar voy a definir las funciones que voy a utilizar para resolver
# cada apartado.

#q = variable de posición
#dq0 = valor inicial de la derivada de q
#d = granularidad del parÃ¡metro temporal
def deriv(q,dq0,d):
   dq = (q[1:len(q)]-q[0:(len(q)-1)])/d
   dq = np.insert(dq,0,dq0)
   return dq

#Ecuacion de Hamilton-Jacobi del oscilador no lineal
def F(q):
    a = 3
    b = 1/2
    ddq = - (8/a)*q*(q**2 -b)
    return ddq

#Resolucion de la ecuacion dinamica ddq = F(q), obteniendo la orbita q(t)
#Los valores iniciales son la posiciÃ³n q0 := q(0) y la derivada dq0 := dq(0)
def orb(n, q0, dq0, F, d):
    q = np.empty([n+1])
    q[0] = q0
    q[1] = q0 + dq0*d
    for i in np.arange(2,n+1):
        q[i] = - q[i-2] + d**2*F(q[i-2]) + 2*q[i-1]
    return q

#Funcion que va a graficar cada diagrama de fases (q,p) respecto a unas
#determinadas condiones iniciales
def simplectica(q0, dq0, F, col=0, d = 10**(-4), marker='-'): 
    n = int(16/d)
    q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
    dq = deriv(q,dq0=dq0,d=d)
    p = dq/2
    plt.plot(q, p, marker, c=plt.get_cmap("winter")(col))

######################
##### APARTADO 1 #####
######################
    
horiz = 12
# Tomo delta como el punto medio del intervalo donde está definido
d = (10**(-3)-10**(-4))/2

fig = plt.figure(figsize=(8,5))
fig.subplots_adjust(hspace=0.4, wspace=0.2)
ax = fig.add_subplot(1,1,1)

#Condiciones iniciales D0 = [0,1] x [0,1] :
#Defino un conjunto de condiciones iniciales que pertenece a D0
#Como p = dq/2, entonces dq0 := p0*2
#Las sucesiones seq_q0 y seq_dq0 tienen un total de 11 puntos, por lo tanto,
#voy a graficar 11*11=121 orbitas finales diferentes

seq_q0 = np.linspace(0.,1.,num=11)
seq_dq0 = np.linspace(0.,1.,num=11)*2
for i in range(len(seq_q0)):
    for j in range(len(seq_dq0)):
        q0 = seq_q0[i]
        dq0 = seq_dq0[j]
        col = (1+i+j*(len(seq_q0)))/(len(seq_q0)*len(seq_dq0))
        #ax = fig.add_subplot(len(seq_q0), len(seq_dq0), 1+i+j*(len(seq_q0)))
        simplectica(q0=q0, dq0=dq0, F=F, col=col, marker='ro', d=d)
ax.set_xlabel("Posición q(t)", fontsize=12)
ax.set_ylabel("Cantidad de movimiento p(t)", fontsize=12)
plt.title('Espacio fásico')
#fig.savefig('Espacio fasico.jpg', dpi=250)
plt.show()

######################
##### APARTADO 2 #####
######################

#Para t=1/4 , es decir, para horiz=1/4, calculamos el area de la envoltura
#convexa del conjunto de puntos p y q. Dichos puntos son una sucesion de puntos
#calculados a partir de unas condiciones iniciales y la siguiente formula:
#    q[i] = - q[i-2] + d**2*F(q[i-2]) + 2*q[i-1]

#Calculamos dicha Ã¡rea para delta = 10**(-3) y para delta = 10**(-4), con el
#objetivo de estimar el intervalo de error.
t = 0.25

def area_t(delta):
    n = int(t/delta)
    # Primero calculamos el diagrama de fases para las condiciones iniciales 
    # definidas por seq_q0 y seq_dq0
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    seq_q0 = np.linspace(0.,1.,num=21)
    seq_dq0 = np.linspace(0.,1.,num=21)*2
    qX = np.array([])
    pX = np.array([])
    for i in range(len(seq_q0)):
        for j in range(len(seq_dq0)):
            q0 = seq_q0[i]
            dq0 = seq_dq0[j]
            q = orb(n,q0=q0,dq0=dq0,F=F,d=delta)
            dq = deriv(q,dq0=dq0,d=delta)
            p = dq/2
            qX = np.append(qX,q[-1])
            pX = np.append(pX,p[-1])
            plt.xlim(-2.2, 2.2)
            plt.ylim(-1.2, 1.2)
            plt.rcParams["legend.markerscale"] = 6
            plt.plot(q[-1], p[-1], marker="o", markersize= 10, 
                     markeredgecolor="red",markerfacecolor="red")
    ax.set_xlabel("Posición q(t)", fontsize=12)
    ax.set_ylabel("Cantidad de movimiento p(t)", fontsize=12)
    plt.title('Diagrama de fases para t=1/4')
    plt.show()
    #fig.savefig('Diagrama de fases.jpg', dpi=250)
    
    #Calculamos el area de la envolvente convexa del diagrama de fases que
    # hemos obtenido
    X = ConvexHull(np.array([qX,pX]).T)
    aX = X.volume
    
    #Definimos dos nuevos conjuntos de condiciones iniciales:
    # Y0 = { (seq_q0, seq_dq0) | y = 0 }
    # Z0 = { (seq_q0, seq_dq0) | x = 1}
    
    #Calculamos las envolventes convexas de los conjuntos de puntos de las
    #condiciones iniciales Y0 & Z0
    qY = np.array([])
    pY = np.array([])
    for i in range(len(seq_q0)):
        q0 = seq_q0[i]
        dq0 = 0
        q = orb(n,q0=q0,dq0=dq0,F=F,d=delta)
        dq = deriv(q,dq0=dq0,d=delta)
        p = dq/2
        qY = np.append(qY,q[-1])
        pY = np.append(pY,p[-1])
    Y = ConvexHull(np.array([qY,pY]).T)
    aY = Y.volume
    
    qZ = np.array([])
    pZ = np.array([])
    for i in range(len(seq_dq0)):
        q0 = 1
        dq0 = seq_dq0[i]
        q = orb(n,q0=q0,dq0=dq0,F=F,d=delta)
        dq = deriv(q,dq0=dq0,d=delta)
        p = dq/2
        qZ = np.append(qZ,q[-1])
        pZ = np.append(pZ,p[-1])
    Z = ConvexHull(np.array([qZ,pZ]).T)
    aZ = Z.volume
    
    #Observando la gráfica, nos damos cuenta que el area de la envolvente
    #convexa X NO es el area real que queremos calcular, por ello tenemos que
    #restarle el área de las envolventes convexas Y & Z
    #Devolvemos el valor del area real del diagrama de fases para t = 1/4
    return aX - aY - aZ

#Definimos a1 y a2 como el area del diagrama de fases para t = 1/4
# cuando delta = 10**(-3) y 10**(-4) respectivamente
a1 = area_t(10**(-3))
a2 = area_t(10**(-4))

print('El área del espacio fásico para t = 1/4 y delta = 10**(-3) es:', a1)
print('El área del espacio fásico para t = 1/4 y delta = 10**(-4) es:', a2)
print("Intervalo de error del valor del Área:", format(a1 - a2,'.3E'))

######################
##### APARTADO 3 #####
######################

fig, ax = plt.subplots()

def animate(t):
    seq_q0 = np.linspace(0.,1.,num=16)
    seq_dq0 = np.linspace(0.,1.,num=16)*2
    d = 10**(-4)
    n = int(t/d)+1 #Sumo 1 para evitar que n sea 0 cuando t=0
    ax.clear()
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    
    for i in range(len(seq_q0)):
        for j in range(len(seq_dq0)):
            q0 = seq_q0[i]
            dq0 = seq_dq0[j]
            q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
            dq = deriv(q,dq0=dq0,d=d)
            p = dq/2
            plt.rcParams["legend.markerscale"] = 6
            ax.set_xlabel("PosiciÃ³n q(t)", fontsize=12)
            ax.set_ylabel("Cantidad de movimiento p(t)", fontsize=12)
            ax.plot(q[-1], p[-1], marker="o", markersize= 10, 
                    markeredgecolor="purple",markerfacecolor="purple")
    return ax,

def init():
    return animate(0),

ani = animation.FuncAnimation(fig, animate, np.linspace(0.,5.,num=16), 
                              init_func=init, interval=100)
#ani.save("animacionPractica3.gif", fps = 5)
