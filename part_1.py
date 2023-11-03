import matplotlib.pyplot as plot
import numpy as np

def genY(X,W):
    return X*W[1]+W[0]

def descensoGradiente(X,Y,W,alpha):
    sumaW0 = 0
    sumaW1 = 0
    print(f"W antes de minimizar {W}")
    #print(f"Y: {Y}")
    for j in range(0,len(Y)):
        sumaW0 = sumaW0 + (Y[j] - (W[1]*X[j]+W[0]))
        sumaW1 = sumaW1 + (Y[j] - (W[1]*X[j]+W[0]))*X[j] 
    
#    for i in range(0,len(W)): 
    n = X.shape[0]
    
    #W[0] = W[0] - alpha*(-2*np.sum((Y-(W[1]*X+W[0]))))/n
    #W[1] = W[1] - alpha*(-2*np.sum(X*(Y-(W[1]*X+W[0]))))/n
    #b_gradient = -2 * np.sum(Y - m*X + b) / n
    #m_gradient = -2 * np.sum(X*(Y - (m*X + b))) / n
    W[0] = W[0] - alpha*sumaW0
    W[1] = W[1] - alpha*sumaW1
    print((-2*np.sum(X*(Y-(W[1]*X+W[0]))))/n)
    return W

def main():
    print("UWU")
    X = []
    #x = np.linspace(1,10,10)
    #y = np.linspace(1,10,10)
    # f(x) = 4x+3
    #for i in range(0,length):
    #    y.append(np.random.randint(0,100))
    #for i in range(0,length):
    np.random.seed(42)
    X = np.array(sorted(list(range(5))*200)) + np.random.normal(size=1000, scale=0.5)
    Y = np.array(sorted(list(range(5))*200)) + np.random.normal(size=1000, scale=0.5)
    X[0] = 1 
    #y = np.random.default_rng().integers(100,size=100)
    #x.append(np.random.Generator.normal(0,1,100))
    #x = [1,2,6]
    #w = [3,4,3]
    #print(f"x: {X}")
    #print(f"Y: {Y}")
    #W = np.random.default_rng().integers(100,size=2)
    W = [1,5]
    alpha = 0.00001
    """ Y = []
    for i in range(0,len(X)):
        Y.append(genY(X[i],W))
    print(Y)  """
    #print(W)
    epsilon = 0.000000015
    n = 0
    print(f"W: {W}")
    while n < 1000:
        W = descensoGradiente(X,Y,W,alpha)
        Y_pred = genY(X,W)
        error = (Y-Y_pred)**2
        if error.all() < epsilon:
            break
        n = n+1
    print(f"W: {W}")
    #Y_pred = genY(X,W)
    plot.scatter(X,Y,color='black')
    plot.plot(X,Y_pred,color='red')
    #plot.plot(error,range(0,1000),color='yellow')
    plot.xlabel("eje x")
    plot.ylabel("Eje y")
    #lot.show()
    #print(Y)
    #print(Y_pred)
    #X.sort()
    #Y_pred.sort()
    
    plot.show()


main()
