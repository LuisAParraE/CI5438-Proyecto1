import random
import matplotlib.pyplot as plot
import numpy as np

"""
Función para generar a Y dados X y W
"""
def genY(X,W):
    return X[1]*W[2]+X[0]*W[1]+W[0]

"""
Función de descenso de gradiente implementada usando la derivada parcial
Recibe una lista de vectores de X
Un vector Y
Una lista de valores W para los pesos
Un alpha
"""
def descensoGradiente(X,Y,W,alpha):
    n = len(X)
    W[0] = W[0] + alpha*(2*np.sum((Y-(W[1]*X[0]+W[0]))))/n
    W[1] = W[1] + alpha*(2*np.sum(X[1]*(Y-(W[1]*X[1]+W[0]))))/n
    W[2] = W[2] + alpha*(2*np.sum(X[2]*(Y-(W[2]*X[2]+W[1]*X[2]+W[0]))))/n
    return W
""""
Función main
Los rngX son inicializadores de valores aleatorios de Numpy con seeds predefinidos
X1,X2,X3 son los vectores de los valores de X
W es una lista de valores aleatorios obtenidos de X1,X2 y X3. 
"""
def main():
    rng1 = np.random.default_rng(seed=25)
    rng2 = np.random.default_rng(seed=30)
    rng3 = np.random.default_rng(seed=35)
    rng4 = np.random.default_rng(seed=40)
    X1 = np.array(sorted(list(range(5))*200)) + rng1.normal(size=1000, scale=0.5)
    X2 = np.array(sorted(list(range(5))*200)) + rng2.normal(size=1000, scale=0.5)
    X3 = np.array(sorted(list(range(5))*200)) + rng3.normal(size=1000, scale=0.5)
    X = [X1,X2,X3]
    W = []
    W.append(random.choice(X1))
    W.append(random.choice(X2))
    W.append(random.choice(X3))
    Y = np.array(sorted(list(range(5))*200)) + rng4.normal(size=1000, scale=0.5)
    alpha = 0.000001
    epsilon = 0.463
    n = 0
    print(f"W: {W}")
    error = []
    while n < 50000:
        W = descensoGradiente(X,Y,W,alpha)
        Y_pred = genY(X,W)
        error.append(sum((Y-Y_pred)**2)/len(Y))
        print(error[-1])
        if error[-1] < epsilon:
            print("Convergió")
            break
        n = n+1
        print(n)
    print(f"W: {W}")
    plot.scatter(X[0],Y,color='black')
    plot.scatter(X[1],Y,color='purple')
    plot.scatter(X[2],Y,color='pink')
    plot.plot(X[0],genY(X,W),color='red')
    plot.xlabel("Eje X")
    plot.ylabel("Eje Y")
    ####
    plot.show()
    plot.plot(range(len(error)),error,color='blue')
    plot.xlabel("Iteraciones")
    plot.ylabel("Error")
    plot.show()


main()
