import random
import matplotlib.pyplot as plot
import numpy as np
from sklearn import preprocessing

def genY(X,W):
    return X*W[1]+W[0]

def descensoGradiente(X,Y,W,alpha):
    n = len(X)    
    W[0] = W[0] + alpha*(2*np.sum((Y-(W[1]*X+W[0]))))/n
    W[1] = W[1] + alpha*(2*np.sum(X*(Y-(W[1]*X+W[0]))))/n
    return W

def main():
    rng = np.random.default_rng()
    X = np.array(sorted(list(range(5))*200)) + rng.normal(size=1000, scale=0.5)
    Y = np.array(sorted(list(range(5))*200)) + rng.normal(size=1000, scale=0.5)
    print(X) 
    W = []
    W.append(random.choice(X))
    W.append(random.choice(X))
    alpha = 0.015
    epsilon = 0.47
    n = 0
    print(f"W: {W}")
    error = []
    while n < 100000:
        W = descensoGradiente(X,Y,W,alpha)
        Y_pred = genY(X,W)
        error.append(sum((Y-Y_pred)**2)/len(Y))
        print(error[-1])
        if error[-1] < epsilon:
            print("Holis")
            break
        n = n+1
        print(n)
    print(f"W: {W}")
    plot.scatter(X,Y,color='black')
    plot.plot(X,Y_pred,color='red')
    plot.xlabel("Eje X")
    plot.ylabel("Eje Y")
    ####
    plot.show()
    plot.plot(range(len(error)),error,color='blue')
    plot.xlabel("Iteraciones")
    plot.ylabel("Error")
    plot.show()


main()
