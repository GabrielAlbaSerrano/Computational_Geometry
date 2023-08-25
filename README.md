# Computational_Geometry
1. Huffman Code and Shannon's First Theorem.
   Through the implementation of the Huffman coding algorithm in Python and its corresponding decoding algorithm, we have been able to       verify that this algorithm is an optimal data compression system. It reduces the lengths of bits by approximately half compared to        other encodings, such as the usual binary encoding. This is a direct consequence of Shannon's First Theorem, which relates these         lengths (specifically, the weighted average of all of them) with entropy, which can be interpreted as the number of independent           variables in a system S, used to construct the parts of that system P(S).
2. Voronoi Diagram and Clustering (particularly interesting).
  The process of grouping or clustering a system involves grouping objects based on similarity into sets or clusters, where members of   the same group share similar characteristics. The appropriate algorithm for such clustering depends on the dataset being analyzed and the intended use of the results.
The concept of grouping similar data is incomplete and subjective, leading to the existence of numerous algorithms. Some well-known models are based on connectivity, centroids, distribution, density, etc. In this practice, we will study two algorithms: one based on centroid-based clustering (KMeans) and another based on density-based clustering (DBSCAN).
The main characteristic of the KMeans algorithm is that it classifies data into clusters of approximately similar size (with a similar number of elements) because it always assigns each element to the nearest centroid based on the corresponding metric. This often results in incorrect cuts at the edges of clusters since the algorithm optimizes centroids, not boundaries. In this case, clusters are known as Voronoi diagrams.
On the other hand, the main characteristic of DBSCAN is that clusters are defined as areas of higher density compared to the rest of the dataset. Objects in sparse areas are known as noise or boundary points. This algorithm doesn't provide good clustering in a system that doesn't have density drops in its elements, meaning a system with similar density across all its regions.

5. Discretization of Continuous Dynamic Systems and Liouville's Theorem.
Discretizing the dynamic system of a nonlinear oscillator allows us to calculate the area of the phase diagram of that system. The algorithm used to calculate the convex hull of the phase diagram verifies Liouville's Theorem, which states that the area of the phase diagram remains invariant over time (as observed in the animation 'p3.gif' generated using Python).

6. Affine Isometric Transformation.
An affine isometric transformation is a type of geometric application that preserves the shape and size of objects while moving them in space. Different affine isometric transformations can be generalized as the composition of a rotation with a translation.
In an affine isometric transformation, distances between points of the object and angles between its lines are preserved. Furthermore, affine isometric transformations are linear, meaning they can be represented using matrices and vectors.
Using Python, I have generated animations 'p4i.gif' and 'p4ii.gif' where you can visually confirm that affine isometric transformations keep the shape and size of objects invariant.

----------------------------------------

# Geometría Computacional

1. Código Huffaman y Primer Teorema de Shannon.
A través de la implementación en Python del algoritmo de codificación de
Huffman y su correspondiente algoritmo de decodificación hemos podido comprobar que 
dicho algoritmo es un óptimo sistema de compresión de datos, ya que reduce, 
aproximadamente, a la mitad las longitudes de bits frente a otras codificaciones,
como la binaria usual. Esto es una consecuencia directa del Primer Teorema de
Shannon, que relaciona dichas longitudes (en concreto la media ponderada de todas ellas)
con la entropía, la cuál puede interpretarse como el número de variables
independientes de un sistema S, que se usan para construir las partes de
dicho sistema P(S).

2. Diagrama de Voronói y Clustering (especialmente interesante).
La clasificación de grupos o clustering de un sistema consiste en agrupar objetos por similitud, en
grupos o conjuntos de manera que los miembros del mismo grupo tengan características similares. El
algoritmo apropiado para dicha clasificación depende del conjunto de datos que se analiza y el uso que
se le dará a los resultados.
La idea de un grupo de datos similares resulta incompleta y subjetiva, y por ello existen miles de
algoritmos. Alguno de los modelos más conocidos se basan en la conectividad, centroides, distribución,
densidad, etc. En esta práctica vamos a estudiar dos algoritmos, uno basado en agrupamiento por
centroides (KMeans) y otro basado en agrupamiento por densidad (DBSCAN).
La principal característica del algoritmo KMeans es que clasifica en clusters de aproximadamente
tamaño similar (con un número similar de elementos), debido a que siempre asignará cada elemento
al centroide más cercano, según la correspondiente métrica. Esto a menudo provoca cortes incorrectos
en los bordes de los grupos, ya que el algoritmo optimiza los centroides, no las fronteras. Los clusters
en este caso se denominan Diagramas de Voronói.
Por otro lado la principal característica del DBSCAN es que los clusters están definidos como áreas de
densidad más alta que en el resto del conjunto de datos. Los objetos en áreas esparcidas son conocidos
como ruido o puntos frontera. Este algoritmo no proporciona un buen clustering en un sistema que no
tiene bajadas de densidad de sus elementos, es decir, un sistema con una densidad similar en todas sus
regiones.

3. Discretización de sistemas dinámicos continuos y Teorema de Liouville.
La discretización del sistema dinámico de un oscilador no lineal, nos permite calcular el área del
diagrama de fases de dicho sistema. El algoritmo utilizado para calcular la envolvente convexa del
diagrama de fases verifica el Teorema de Liouville que afirma que el área del diagrama de fases es
invariante respecto al tiempo (se puede observar en la animación 'p3.gif' generado a través de Python.

4. Transformación isométrica afín.
La transformación isométrica afín es un tipo de aplicación geométrica que preserva la forma y el
tamaño de los objetos mientras los mueve en el espacio. Las diferentes transformaciones isométricas
afines se pueden generalizar como la composición de una rotación junto con una traslación.
En una transformación isométrica afín, se mantienen las distancias entre los puntos del objeto, así
como los ángulos entre las líneas que lo componen. Además, las transformaciones isométricas y afines
son lineales, lo que significa que se pueden representar por medio de matrices y vectores.
Utilizando Python he generado las animaciones 'p4i.gif' y 'p4ii.gif' donde se pueden comprobar
visualmente que las transformaciones isométrica afines mantienen invariantes la forma y tamaño de los
objetos.
