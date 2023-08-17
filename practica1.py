"""
Práctica 1. Código de Huffmann y Teorema de Shannon
"""

import os
import numpy as np
import pandas as pd

#### Vamos al directorio de trabajo
os.getcwd()
#os.chdir(ruta)
#files = os.listdir(ruta)

with open('GCOM2023_pract1_auxiliar_eng.txt', 'r',encoding="utf8") as file:
      en = file.read()
     
with open('GCOM2023_pract1_auxiliar_esp.txt', 'r',encoding="utf8") as file:
      es = file.read()


#### Contamos cuantos caracteres hay en cada texto
from collections import Counter
tab_en = Counter(en)
tab_es = Counter(es)

#### Transformamos en formato array de los carácteres (states) y su frecuencia
#### Finalmente realizamos un DataFrame con Pandas y ordenamos con 'sort'
tab_en_states = np.array(list(tab_en))
tab_en_weights = np.array(list(tab_en.values()))
tab_en_probab = tab_en_weights/float(np.sum(tab_en_weights))
distr_en = pd.DataFrame({'states': tab_en_states, 'probab': tab_en_probab})
distr_en = distr_en.sort_values(by='probab', ascending=True)
distr_en.index = np.arange(0,len(tab_en_states))

tab_es_states = np.array(list(tab_es))
tab_es_weights = np.array(list(tab_es.values()))
tab_es_probab = tab_es_weights/float(np.sum(tab_es_weights))
distr_es = pd.DataFrame({'states': tab_es_states, 'probab': tab_es_probab })
distr_es = distr_es.sort_values(by='probab', ascending=True)
distr_es.index = np.arange(0,len(tab_es_states))

##### Para obtener una rama, fusionamos los dos states con menor frecuencia
distr = distr_en
''.join(distr['states'][[0,1]])

### Es decir:
states = np.array(distr['states'])
probab = np.array(distr['probab'])
state_new = np.array([''.join(states[[0,1]])])   #Ojo con: state_new.ndim
probab_new = np.array([np.sum(probab[[0,1]])])   #Ojo con: probab_new.ndim
codigo = np.array([{states[0]: 0, states[1]: 1}])
states = np.concatenate((states[np.arange(2,len(states))], state_new), axis=0)
probab = np.concatenate((probab[np.arange(2,len(probab))], probab_new), axis=0)
distr = pd.DataFrame({'states': states, 'probab': probab, })
distr = distr.sort_values(by='probab', ascending=True)
distr.index=np.arange(0,len(states))

#Creamos un diccionario
branch = {'distr':distr, 'codigo':codigo}

## Ahora definimos una función que haga exáctamente lo mismo
def huffman_branch(distr):
    states = np.array(distr['states'])
    probab = np.array(distr['probab'])
    state_new = np.array([''.join(states[[0,1]])])
    probab_new = np.array([np.sum(probab[[0,1]])])
    codigo = np.array([{states[0]: 0, states[1]: 1}])
    states = np.concatenate((states[np.arange(2,len(states))], state_new), axis=0)
    probab = np.concatenate((probab[np.arange(2,len(probab))], probab_new), axis=0)
    distr = pd.DataFrame({'states': states, 'probab': probab})
    distr = distr.sort_values(by='probab', ascending=True)
    distr.index=np.arange(0,len(states))
    branch = {'distr':distr, 'codigo':codigo}
    return(branch) 

def huffman_tree(distr):
    tree = np.array([])
    while len(distr) > 1:
        branch = huffman_branch(distr)
        distr = branch['distr']
        code = np.array([branch['codigo']])
        tree = np.concatenate((tree, code), axis=None)
    return(tree)
 
distr = distr_en 
tree = huffman_tree(distr)
tree[0].items()
tree[0].values()

#Buscar cada estado dentro de cada uno de los dos items
list(tree[0].items())[0][1] ## Esto proporciona un '0'
list(tree[0].items())[1][1] ## Esto proporciona un '1'



""" Voy a usar la plantilla del campus virtual como base para desarrollar 
funciones que resuelvan los apartados de la práctica. """

""" Voy a crear una función que represente el código binario de cada estado de
un sistema, que se obtiene al aplicar el algoritmo de Huffman. Dicha función
aceptará una lista de estados del sistema y un árbol de Huffman, y devolverá
un diccionario, cuyas keys serán los estados del sistema y sus respectivos 
values sus expresiones en código binario Huffman. 
La idea de la función es ver para cada estado, cuando aparece en los elementos 
del árbol y añadir un 0 ó un 1, dependiendo del respectivo value de la llave en
la que aparece el estado. """

def codigo_huffman(estados,arbol):
    result = {}
    for i in range(len(estados)):
        a = estados[i]
        codigo = ''
        for j in range(len(arbol)):
            if a in list(arbol[j])[0]:
                codigo = '0' + codigo
            if a in list(arbol[j])[1]:
                codigo = '1' + codigo
        result[a] = codigo
    return result

""" Sean a[i] los diferentes estados de un sistema S, w[i] las probabilidades
asociadas a cada estado a[i], y |c[i]| la longitud de cada cadena binaria c[i]
obtenida aplicando el algoritmo de Huffman .
Sea W = suma(w[i]), en particular si w[i] son frecuencias relativas como en
los sistemas SEng y SEsp, entonces W = 1.
Definimos la longitud media de un sistema S:
    L(S) = (1/W) * suma( w[i] * |c[i]| ) 
"""

def longitud_media(w,codigo):
    cadenas = list(codigo.values())
    result = 0
    for i in range(len(w)):
        result += w[i]*len(cadenas[i])
    return result

""" Definimos la entropía de un sistema. """

import math
def entropia_sistema(p): # p es una lista de las probabilidades de cada estado
    H = 0
    for i in range(len(p)):
        H -= p[i] * math.log2(p[i])
    return H

""" Para codificar una palabra, ó, varias palabras, concatenamos la
representación binaria de cada caracter (estado). La función codificar acepta 
como argumentos de entrada una cadena de caracteres (estados) x, y un
diccionario cod, cuyos índices son los estados del sistema, y sus respectivos
valores sus representaciones en binario. """

def codificar(x,cod):
    y = list(x)
    result = ''
    for i in range(len(y)):
        result += cod[y[i]]
    return result

""" Para decodificar una representación en binario de la codificación Huffman,
se busca, empezando por la izquierda, la representación de un estado (caracter
de SEng o SEsp) y se añade a result, hasta que no haya ningún 0, ni ningún 1. """

def decodificar(x,cod):
    estados = list(cod.keys())
    cadenas = list(cod.values())
    y = list(str(x))
    result = ''
    i = 0
    while i<len(y):
        cad = ''
        # Para buscar la representación binaria de un estado vamos añadiendo
        # elementos de x, hasta que coincide con alguna cadena del diccionario cod
        while cad not in cadenas: 
            cad += y[i]
            i += 1
        indice = cadenas.index(cad)
        result += estados[indice]
    return result

""" Finalmente este script va a imprimir por pantalla, las soluciones a las 
preguntas de la práctica. Voy a declarar unas variables globales utilizando el 
comando iloc para DataFrame. Las cuáles son listas que representan los estados
y las probabilidades del sistema SEng (alfabeto inglés), ordenadas en orden
ascendente según la probabilidad de cada estado. Además voy a declarar una
variable que representa el árbol de Huffman que se obtiene con la función de la
plantilla. Análogamente con SEsp (alfabeto español). """

estados_en = list(distr_en.iloc[:, 0])
prob_en = list(distr_en.iloc[:, 1])
arb_en = huffman_tree(distr_en)
estados_es = list(distr_es.iloc[:, 0])
prob_es = list(distr_es.iloc[:, 1])
arb_es = huffman_tree(distr_es)

huffman_en = codigo_huffman( estados_en, arb_en )
cod_en = np.array(list(huffman_en.values()))
huffman_es = codigo_huffman( estados_es, arb_es )
cod_es = np.array(list(huffman_es.values()))
long_en = longitud_media( prob_en, huffman_en)
long_es = longitud_media( prob_es, huffman_es)
entropia_en = entropia_sistema(prob_en)
entropia_es = entropia_sistema(prob_es)
e1 = format( 1/len(estados_en), '.4E')
e2 = format( 1/len(estados_es), '.4E')
dim_en = codificar('dimension', huffman_en)
dim_es = codificar('dimension', huffman_es)
dim_bin = '01100100 01101001 01101101 01100101 01101110 01110011 01101001 01101111 01101110'
x = '0101010001100111001101111000101111110101010001110'
decod_x = decodificar(x,huffman_en)

print('\n ----- APARTADO 1 ----- \n')
print('El código binario Huffman del sistema SEng es: \n')
print(pd.DataFrame({'States': np.array(estados_en), 'Coding': cod_en}))
print('\n')
print('El código binario Huffman del sistema SEsp es: \n')
print(pd.DataFrame({'Estados': np.array(estados_es), 'Codificación': cod_es}))
print('\n')
print('La longitud media de SEng es: ' + str(long_en) + ' ± ' + str(e1))
print('La longitud media de SEsp es: ' + str(long_en) + ' ± ' + str(e2))
print('La entropia H de SEng es: ' + str(entropia_en) + ' ± ' + str(e1))
print('La entropia H de SEsp es: ' + str(entropia_es) + ' ± ' + str(e2) + '\n')
print('Observamos que se satisface el Primer Teorema de Shannon porque:')
print('   H(S) <= L(S) < H(S)+1 , para S = SEng, SEsp \n')

print('\n ----- APARTADO 2 ----- \n')
print('La codificación de la palabra "dimension" en SEng es: ' + dim_en)
print('La codificación de la palabra "dimension" en SEsp es: ' + dim_es)
print('La codificación en código binario usual es: ' + dim_bin + '\n')
print('Observamos que la codificación en código binario usual es más extensa que el código binario Huffman, tanto de SEng como de SEsp. En particular es ' + str( format(len(dim_bin)/len(dim_en),'.4E') ) + ' veces más extensa.\n')

print('\n ----- APARTADO 3 ----- \n')
print('La decodificación de ' + x + ' es: ' + decod_x)


print('\n ----- EXTRA ----- \n')
print('Para comprobar el correcto funcionamiento de la codificación Huffman, vamos a testear si funcionan correctamente las funciones codificar y decodificar. ')

print('Test para SEng:')
test_en = ['computational geometry', 'logistic attractor', 'convex hull', 'nonlinear dynamic system', 'chaos', 'devaney', 'exponente', 'lyapunov', 'fractal dimension', 'hausdorff']
errores_en = 0
for k in test_en:
    codificacion = codificar(k, huffman_es)
    if k == decodificar( codificacion, huffman_es ):
        print('La cadena de caracteres "' + k + '" se ha codificado correctamente.')
    else:
        print('La cadena de caracteres ' + k + ' no se ha codificado correctamente.')
        errores_en += 1
if errores_en == 0:
    print('No hay errores al aplicar el test. Éxito!!! \n')
else:
    print('En total, al aplicar el test, hemos encontrado ' + errores_en + ' errores. \n')


print('Test para SEsp:')
test_es = ['geometria computacional', 'atractor logistico', 'envolvente convexa', 'sistema dinamico no lineal', 'caos', 'devaney', 'exponente', 'lyapunov', 'medidas fractales', 'hausdorff']
errores_es = 0
for k in test_es:
    codificacion = codificar(k, huffman_es)
    if k == decodificar( codificacion, huffman_es ):
        print('La cadena de caracteres "' + k + '" se ha codificado correctamente.')
    else:
        print('La cadena de caracteres ' + k + ' no se ha codificado correctamente.')
        errores_es += 1
if errores_es == 0:
    print('No hay errores al aplicar el test. Éxito!!! \n')
else:
    print('En total, al aplicar el test, hemos encontrado ' + errores_es + ' errores. \n')    

print('Observaciones: ')
print('(1) Hay que tener en cuenta que SEng y SEsp no corresponden a los alfabetos completos de sus respectivos idiomas, sino a la muestra de los archivos proporcionados en el campus virtual, por lo tanto no podremos codificar todas las palabras del inglés y el español.')
print('(2) Si codificamos un string en SEsp, al decodificarlo en SEng, obtendremos un error, y viceversa.')
