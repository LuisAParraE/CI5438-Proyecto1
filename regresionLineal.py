import pandas
import numpy
import matplotlib.pyplot as plt

#Aca se encuentran todas las variables dummy para introducir como datos de entrada
dummyMake = ["Audi", "BMW", "Chevrolet","Datsun", "Ferrari","Fiat", "Ford", "Honda", "Hyundai", "Isuzu",
             "Jaguar", "Jeep", "Kia", "Lamborghini", "Land Rover", "Lexus", "Mahindra", "Maruti Suzuki",
             "Maserati", "Mercedes-Benz", "MG", "MINI", "Mitsubishi", "Nissan", "Porche", "Renault", "Rolls-Royce",
             "Skoda","Ssangyong", "Tata", "Toyota", "Volkswagen", "Volvo"]

dummyFuelType = ["CNG", "Diesel","Electric","Hybrid", "LPG", "Petrol"]

dummyTransmision = ["Manual", "Automatic"]

dummyOwner = ["First", "Second", "Third", "Fourth", "UnRegistered Car"]

dummySellerType = ["Individual", "Corporate"]

dummySeats = [2,4,5,6,7,8]

#Pesos que usara la regresión final
weights = []

#La perdida por iteración
loss = []

#Función para agarrar la data cruda leida por pandas y procesarla a algo utilizable
def formatDataset(dataset):
    clearDataset = []

    #Buscamos el mayor y menor de ciertas columnas para poder parametrizarlas
    minYear = dataset["Year"].max()
    maxYear = dataset["Year"].min()

    maxKm = dataset["Kilometer"].max()
    minKm = dataset["Kilometer"].min()

    maxGasTank = dataset["Fuel Tank Capacity"].max()
    minGasTank = dataset["Fuel Tank Capacity"].min()

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
        normalizeYear = (rawRow["Year"] - minYear)/(maxYear-minYear)
        newRow.append(normalizeYear)
        
        #Normalizamos los Km
        normalizeKm = (rawRow["Kilometer"] - minKm)/(maxKm-minKm)
        newRow.append(normalizeKm)

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
        
        #Verificamos las variables dummy del tipo de vendedor
        for seller in dummySellerType:
            if rawRow["Seller Type"] == seller:
                newRow.append(1)
            else:
                newRow.append(0)

        #Verificamos las variables dummy de la cantidad de asientos
        for seats in dummySeats:
            if rawRow["Seating Capacity"] == seats or (numpy.isnan(rawRow["Seating Capacity"]) and (5 ==seats)):
                newRow.append(1)
            else:
                newRow.append(0)

        #Normalizamos la capacidad del tanque
        if numpy.isnan(rawRow["Fuel Tank Capacity"]):
            normalizeTank = (35 - minGasTank)/(maxGasTank-minGasTank)
            newRow.append(normalizeTank)
        else:
            normalizeTank = (rawRow["Fuel Tank Capacity"] - minGasTank)/(maxGasTank-minGasTank)
            newRow.append(normalizeTank)
        
        clearDataset.append(newRow)

    return clearDataset

#Función para entrenar la data, si los pesos no estan inicializados, se inicializan
def train(dataset,results,alpha,n):

    #Inicialización de los pesos
    global weights
    if len(weights)-1 != len(dataset[0]):
        weights = []

        for w in range(0, len(dataset[0])+1):
            weights.append(0)

    #Actualizacion de los pesos
    i = 0
    while (i<n):
        newLoss = 0
        print(i)

        #Aca se actualizan los pesos
        for j in range(1,len(weights)):
            for k in range(0,len(dataset)):
                weights[j] = weights[j] + alpha * dataset[k][j-1]*(results.iloc[k] - linearRegression(dataset[k]))

        #Aca se calcula la perdida
        for j in range(0,len(dataset)):
            newLoss = newLoss + (results.iloc[j] - linearRegression(dataset[j]))**2

        loss.append(newLoss)
        i = i+1


#Función Base de regresión para multiples variables
def linearRegression(values):
    global weights

    h = 1*weights[0] 

    for i in range(0,len(values)):
        h = h + weights[i+1]*values[i]

    return h

def main():
    #Numero de iteraciones
    n = 100
    #Se lee el CSV con pandas
    df = pandas.read_csv('CarDekho.csv')
    
    #Se procesan los datos para tener valores interactuables
    clearDataset = formatDataset(df)

    #Entrenamos el modelo
    train(clearDataset,df["Price"],0.00001,n)

    plt.plot(range(1,n+1),loss)
    plt.ylabel("Perdida")
    plt.xlabel("n Iteraciones")
    plt.show()

    #for i in range(0,20):
    #    print(linearRegression(clearDataset[i]))

main()