import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#Aca se encuentran todas las variables dummy para introducir como datos de entrada
dummyMake = ["Audi", "BMW", "Chevrolet","Datsun", "Ferrari","Fiat", "Ford", "Honda", "Hyundai", "Isuzu",
             "Jaguar", "Jeep", "Kia", "Lamborghini", "Land Rover", "Lexus", "Mahindra", "Maruti Suzuki",
             "Maserati", "Mercedes-Benz", "MG", "MINI", "Mitsubishi", "Nissan", "Porche", "Renault", "Rolls-Royce",
             "Skoda","Ssangyong", "Tata", "Toyota", "Volkswagen", "Volvo"]

dummyFuelType = ["CNG", "Diesel","Electric","Hybrid", "LPG", "Petrol"]

dummyTransmision = ["Manual", "Automatic"]

dummyOwner = ["First", "Second", "Third", "Fourth", "UnRegistered Car"]

dummySeats = [2,4,5,6,7,8]

#Pesos que usara la regresión final
weights = []
bestWeights = []

#La perdida por iteración en el entrenamiento
trainLoss = []
meanTrainLoss = []
lessTrainLoss = 200

#Función para agarrar la data cruda leida por pandas y procesarla a algo utilizable
def formatDataset(dataset):
    clearDataset = []
    clearAnswer = []

    #Llenamos los valores nulos con la Moda
    dataset["Seating Capacity"] = dataset["Seating Capacity"].fillna(dataset["Seating Capacity"].mode()[0])
    dataset["Fuel Tank Capacity"] = dataset["Fuel Tank Capacity"].fillna(dataset["Fuel Tank Capacity"].mode()[0])

    #Normalizamos los valores
    dataset["Year"] = (dataset["Year"] - dataset["Year"].min()) / (dataset["Year"].max() - dataset["Year"].min())
    dataset["Kilometer"] = (dataset["Kilometer"] - dataset["Kilometer"].min()) / (dataset["Kilometer"].max() - dataset["Kilometer"].min())
    dataset["Fuel Tank Capacity"] = (dataset["Fuel Tank Capacity"] - dataset["Fuel Tank Capacity"].min()) / (dataset["Fuel Tank Capacity"].max() - dataset["Fuel Tank Capacity"].min())

    #Este ciclo va por todas las filas del data set
    for i in range(0,len(dataset.index)):
        newRow = []
        rawRow = dataset.iloc[i]

        #Verificamos las variables dummy de la marca
        for brand in dummyMake:
            if rawRow["Make"] == brand:
                newRow.append(1)
            else:
                newRow.append(0)

        #Normalizamos el año
        newRow.append(rawRow["Year"])
        
        #Normalizamos los Km
        newRow.append(rawRow["Kilometer"])

        #Verificamos las variables dummy del tipo de gasolina
        for gasType in dummyFuelType:
            if rawRow["Fuel Type"] == gasType:
                newRow.append(1)
            else:
                newRow.append(0)

        #Verificamos las variables dummy de la transmisión
        for transmission in dummyTransmision:
            if rawRow["Transmission"] == transmission:
                newRow.append(1)
            else:
                newRow.append(0)

        #Verificamos las variables dummy del dueño
        for owner in dummyOwner:
            if rawRow["Owner"] == owner:
                newRow.append(1)
            else:
                newRow.append(0)

        #Verificamos las variables dummy de la cantidad de asientos
        for seats in dummySeats:
            if rawRow["Seating Capacity"] == seats:
                newRow.append(1)
            else:
                newRow.append(0)

        #Normalizamos la capacidad del tanque
        newRow.append(rawRow["Fuel Tank Capacity"])
        
        clearAnswer.append(rawRow["Price"])
        clearDataset.append(newRow)

    return clearDataset,clearAnswer

#Función para entrenar la data, si los pesos no estan inicializados, se inicializan
def train(dataset,results,alpha,n):

    #Inicialización de los pesos
    global weights, bestWeights,lessTrainLoss
    
    weights = []

    for w in range(0, len(dataset[0])):
        weights.append(0)

    #Actualizacion de los pesos
    i = 0
    while (i<n):
        newtrainLoss = 0
        print(i)

        #Aca se actualizan los pesos
        for j in range(0,len(weights)):
            add = 0
            for k in range(0,len(dataset)):
                add = add + dataset[k][j] * (results[k] - linearRegression(dataset[k]))     
            weights[j] = weights[j] + alpha * add
        
        #Aca se calcula la perdida
        for j in range(0,len(dataset)):
            newtrainLoss = newtrainLoss + (results[j] - linearRegression(dataset[j]))**2
        trainLoss.append(newtrainLoss)
        meanTrainLoss.append(newtrainLoss/len(dataset))
        #Guardamos ls mejores pesos con su perdida
        if newtrainLoss < lessTrainLoss:
            lessTrainLoss = newtrainLoss
            bestWeights = weights
        
        i = i+1

#Función Base de regresión para multiples variables
def linearRegression(values):
    global weights

    h = 0
    for i in range(0,len(weights)):
        h = h + weights[i] * values[i]

    return h

def main():
    global lessTrainLoss
    #Numero de iteraciones
    n = 5000
    alpha = 0.001

    #Se lee el CSV con pandas
    df = pandas.read_csv('CarDekho.csv')

    df["Price"] = (df["Price"] - df["Price"].min()) / (df["Price"].max() - df["Price"].min())
    histogram = df.hist(column=["Price", "Year","Seating Capacity", "Fuel Tank Capacity", "Kilometer"],bins=50)
    plt.show()
    
    #boxplot = df.boxplot(column=["Fuel Tank Capacity"])
    #boxplot.plot()
    #plt.show()
    
    #boxplot = df.boxplot(column=["Year"])
    #boxplot.plot()
    #plt.show()
    
    #boxplot = df.boxplot(column=["Kilometer"])
    #boxplot.plot()
    #plt.show()
    #Se procesan los datos para tener valores interactuables
    formatVariables, formatPrice = formatDataset(df)

    #Hacemos el Cross Data Validation
    dataTraining,dataTest,answerTraining,answerTest = train_test_split(
        formatVariables, formatPrice,test_size= 0.2,shuffle=True
    )
    
    #Entrenamos el modelo
    train(dataTraining,answerTraining,alpha,n)
    print(weights)

    plt.plot(range(1,n+1),trainLoss)
    plt.title(f"Loss with alpha:{alpha}")
    plt.axis((0,n,0,15))
    plt.ylabel("Loss")
    plt.xlabel("Iteration Number")
    plt.show()

    plt.plot(range(1,n+1),meanTrainLoss)
    plt.title(f"Mean Loss with alpha:{alpha}")
    plt.axis((0,n,0,0.01))
    plt.ylabel("Mean Loss")
    plt.xlabel("Iteration Number")
    plt.show()

    print(f"menor perdida:{lessTrainLoss}")

main()