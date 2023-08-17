""" 

GCOM - PRÁCTICA 2: DIAGRAMA DE VORONÓI Y CLUSTERING 
NOMBRE: Gabriel Alba Serrano
SUBGRUPO: U1

"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

# Aquí tenemos definido el sistema X de 1500 elementos (personas) con dos 
# estados: X1 = "nivel de estrés" y X2 = "afición al rock"
archivo1 = "C:/Users/Gabriel/Documents/CC-MAT/Geometría Computacional/Práctica 2/Personas_en_la_facultad_matematicas.txt"
archivo2 = "C:/Users/Gabriel/Documents/CC-MAT/Geometría Computacional/Práctica 2/Grados_en_la_facultad_matematicas.txt"
X = np.loadtxt(archivo1)
Y = np.loadtxt(archivo2)
labels_true = Y[:,0]

#Si quisieramos estandarizar los valores del sistema, haríamos:
#from sklearn.preprocessing import StandardScaler
#X = StandardScaler().fit_transform(X)  

#Envolvente convexa, envoltura convexa o cápsula convexa, de X
hull = ConvexHull(X)
convex_hull_plot_2d(hull)
plt.title('Envolvente convexa del sistema X')
plt.plot(X[:,0],X[:,1],'ro', markersize=1)
plt.show()



""" APARTADO 1 """

""" Aplicamos al sistema X el algoritmo KMeans para cada número de vecindades
k = {2,3,...,15}, y comprobamos qué coeficiente de Silhouette es mayor según k,
y de esta forma obtenemos el número óptimo de celdas de Voronoi. """

neighborhoods = list(range(2,16))
silhouette = []
for k in neighborhoods:
    #Usamos la inicialización aleatoria "random_state=0"
    kmeans = KMeans(n_clusters = k, random_state=0).fit(X)
    labels = kmeans.labels_
    silhouette.append( metrics.silhouette_score(X, labels) )

# Gráfica del coeficiente de Silhouette respecto al número de vecindades
plt.plot(neighborhoods,silhouette)
plt.xlabel('Número de vecindades')
plt.ylabel('Coeficiente de Silhouette')
plt.title('K-Means')
plt.savefig('Silhouette KMeans.jpg')
plt.show()

""" Definimos n_clusters := k , tal que max(silhouette) = silhouette[k]
De esta forma, n_clusters es el número de vecindades que mejor empareja los
elementos de cada cluster (vecindad), es decir es el número de vecindades más
óptimo. """
n_clusters = neighborhoods[ silhouette.index(max(silhouette))]
print('El número óptimo de vecindades es: k = ' + str(n_clusters) + '\n')

""" Definimos la etiqueta que indica a qué cluster pertenece cada elemento y 
los centros de cada cluster, cuando el número de vecindades es n_clusters """
kmeans = KMeans(n_clusters = n_clusters, random_state=0).fit(X)
labels = kmeans.labels_ 
centers = kmeans.cluster_centers_

""" Representamos la clasificación del sistema X, por el algoritmo KMeans 
y su respectivo diagrama de Voronoi. """

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

""" Definimos las celdas de Voronoi según los centros de los clusters,y 
graficamos el diagrama de Voronoi. """
vor = Voronoi(centers)
voronoi_plot_2d(vor)

""" Graficamos cada elemento del sistema X, según al cluster al que pertenezcan. """
for lab, col in zip(unique_labels, colors):
    if lab == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == lab)

    xy = X[class_member_mask]
    plt.xlim(min(X[:,0]), max(X[:,0]))
    plt.ylim(min(X[:,1]), max(X[:,1]))
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5)
    
""" Graficamos los centros de cada cluster. """
plt.plot(centers[:,0], centers[:,1], 'o', markersize=12, markerfacecolor="green", label='Centro del cluster')

""" Graficamos los puntos a=(0,0) y b=(0,-1), que los necesitaremos para el 
apartado 3. """
problem = np.array([[0, 0], [0, -1]])
clases_pred = kmeans.predict(problem)
plt.plot(problem[:,0],problem[:,1],'o', markersize=12, markerfacecolor="red", label='A=(0,0) \nB=(0,-1)')

plt.title('Diagrama de Voronói y Clustering KMeans de X con ' + str(n_clusters) + ' clusters')
plt.legend()
plt.savefig('Diagrama de Voronói y Clustering KMeans con 3 clusters.jpg')
plt.show()



""" APARTADO 2 """

""" Aplicamos al sistema X el algoritmo DBSCAN para epsilon (umbral de distancia) 
perteneciente al intervalo (0.1 , 0.4), y comprobamos qué coeficiente de Silhouette 
es mayor según epsilon, y de esta forma obtenemos el número óptimo de celdas de Voronoi. """

# Los posibles valores de epsilon van a ser los siguientes puntos:
epsilon = np.arange(0.1, 0.4, 0.005)


### Métrica 'Euclidean'
s_euclidean = []

for e in epsilon:
    db = DBSCAN(eps=e, min_samples=10, metric='euclidean').fit(X)
    labels = db.labels_
    s = metrics.silhouette_score(X, labels)
    s_euclidean.append(s)

# Graficamos
plt.plot(epsilon, s_euclidean)
plt.xlabel('Umbral de distancia')
plt.ylabel('Coeficiente de Silhouette')
plt.title('DBSCAN Euclidean')
plt.savefig('Silhouette DBSCAN Euclidean.jpg')
plt.show()

# Definimos el epsilon óptimo para la métrica Euclidean
epsilon_euclidean = format(epsilon[ s_euclidean.index(max(s_euclidean)) ] , '.3f')
print('El umbral de distancia óptimo para la métrica Euclidean es: ε = ' + epsilon_euclidean + '\n')


### Métrica 'Manhattan'
s_man = []

for e in epsilon:
    db = DBSCAN(eps=e, min_samples=10, metric='manhattan').fit(X)
    labels = db.labels_
    s = metrics.silhouette_score(X, labels)
    s_man.append(s)

# Graficamos
plt.plot(epsilon, s_man)
plt.xlabel('Umbral de distancia')
plt.ylabel('Coeficiente de Silhouette')
plt.title('DBSCAN Manhattan')
plt.savefig('Silhouette DBSCAN Manhattan.jpg')
plt.show()

# Definimos el epsilon óptimo para la métrica Manhattan
epsilon_man = format( epsilon[ s_man.index(max(s_man)) ] , '.3f' )
print('El umbral de distancia óptimo para la métrica Manhattan es: ε = ' + epsilon_man + '\n')

""" Finalmente en este apartado vamos a comparar gráficamente el resultado de
aplicar el algoritmo DBSCAN con métrica 'Euclidean' y 'Manhattan' al conjunto X,
es decir, el clustering, frente al clustering obtenido al aplicar KMeans """

# Gráfica DBSCAN Euclidean

db = DBSCAN(eps=float(epsilon_euclidean), min_samples=10, metric='euclidean').fit(X)
labels = db.labels_
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

plt.figure(figsize=(8,4))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=3)

plt.title('DBSCAN Euclidean del sistema X con ' + str(n_clusters_) + ' clusters')
plt.show()


# Gráfica DBSCAN Manhattan

db = DBSCAN(eps=float(epsilon_man), min_samples=10, metric='manhattan').fit(X)
labels = db.labels_
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

plt.figure(figsize=(8,4))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=3)

plt.title('DBSCAN Manhattan del sistema X con ' + str(n_clusters_) + ' clusters')
plt.show()

# Conclusión
print('Como conclusión observamos que el algoritmo KMeans clasifica los elementos de X en 3 clusters, frente al algoritmo DBSCAN que con ambas métricas (Euclidean y Manhattan) clasifica los elementos en un único cluster.')
print('Hay que destacar que la clasificación del algoritmo KMeans es más óptima que la del DBSCAN porque su coeficiente de Silhouette es el que más se aproxima a 1. \n')



""" APARTADO 3 """

problem = np.array([[0, 0], [0, -1]])
clases_pred = kmeans.predict(problem)
print('Según el clustering resultante de aplicar el algoritmo KMeans, los puntos a=(0,0) y b=(0,-1) deberían pertenecer a los clusters ' + str(clases_pred[0]) + ' y ' + str(clases_pred[1]) + ', respectivamente.')

