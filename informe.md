# Informe del Proyecto I - CI5438

## Implementación - Parte 1

Se implementó la función de descenso de gradiente para la regresión lineal multivariada, usando como condiciones de convergencia:
- Un número de iteraciones fijo
- Que la función de pérdida cuadrática sea menor a un $\epsilon$ definido pequeño.

Se usó la función dada en clase para el caso monovariado, dado que no fue posible implementar la función multivariada. La regresión lineal usada viene definida por: $h_w(x) = w_0+w_1 \cdot x$.

La función de pérdida cuadrática usada fue $L^2(y,\hat{y}) = (y - \hat{y})^2$, de acuerdo al enunciado del proyecto. 

Dentro del código se definió que la condición de convergencia para las iteraciones fuera de 10.000 iteraciones, mientras que $\epsilon = 0.46$, ya que los valores oscilaban entre $0.45$ y $0.47$ cuando disminuía el error, sin tener muchos cambios cuando alcanzaba la convergencia.

La función definida para hacer las pruebas fue $f(x) = w_0+w_1 \cdot x_1$


## Parte 2 - Preprocesamiento de los datos

Los datos a ser procesados provienen del dataset `CarDekho.csv`. Este dataset está compuesto por diferentes datos sobre automóviles, y se quiere predecir el atributo *Precio* de un automóvil. Se sugirió usar los atributos de entrada:

- Make
- Year
- Kilometer
- Fuel Type
- Transmission
- Owner
- Seating Capacity
- Fuel Tank Capacity


Estos atributos serían las columnas de datos, si bien, existen otros factores que podrían ser influyentes como el torque y la fuerza, no conseguimos una manera correcta de representar ambos números, por otra parte, definirlos como variables categóricas no era viable dada la gran cantidad de variables que surgirían por tomar esta acción. Para los casos de *Make*, *Transmission*, *Seating Capacity*, *Fuel Type* y *Owner* se subdividió en variables dummy, es decir, variables dicotómicas que pueden tomar valores 1 o 0, siendo una variable nueva cada una de los elementos únicos de Make. 

En el caso de *Seating Capacity* se optó por modelarlo como una variable categórica y no contínua, dado que son pocas opciones y solo puede adoptar valores enteros positivos, en el dataset estaba subdivido de 2 a 8.

Para los valores numéricos como *Kilometer*, *Price*, *Fuel Tank Capacity* y *Year* éstos fueron parametrizados, para no tener que trabajar con datos númericos extremadamente altos, principalmente *Kilometer* y *Price*.

Y finalmente para manejar los valores faltantes se decidió utilizar la moda como dato por defecto para que no afectara de sobremanera a la información.

Además de esto se eliminaron los siguientes elementos que se consideraban atípicos:

- Kilometrajes de 2 millones y 950 mil
- Tipos de combustible que solo tenían 1 o 2 casos como *CNG+CNG*,*PETROL + CNG* y *PETROL + LPG*
- Owner se eliminó el caso de *4 or more* ya que no tenía casos significativos.

Se tienen los casos de:

#### Fuel Tank Capacity

El diagrama de caja del atributo *Fuel Tank Capacity* se puede observar que la mayoría de los carros tienen sus valores entre 40 y 60 Litros, pero hay datos atípicos que superan los 85L.

![Fuel Tank Capacity](/images/box_plot_fuel_capacity.png)

#### Year

El caso del atributo *Year*, se observa que la mayoría de los carros datan de entre el 2014 al 2019, pero nuevamente se observan puntos de vehículos previos al 2006, e inclusive un carro de 1988.

![Year](/images/box_plot_year.png)

#### Kilometer

Este diagrama de caja muestra la distancia recorrida por el carro, donde un carro 0Km representa un carro nuevo. El kilometraje varía entre 20000 a 100000Km

![Kilometer](/images/box_plot_kilometer.png)

## Parte 3 - 

Comparativa de Mejor pérdida, para las iteraciones dado un alpha (Tasa de aprendizaje)
 | Iteraciones/Alpha | 0.001 | 0.0001 | 0.00005 | 0.00001 |
  :-----------: |------|-------|-----------|-------- |
 | 100 | 3.118280139 | 4.389778897 | 4.685985727 | 5.992504547|
 | 300 |2.327452545 | 2.985638202| 4.430267141 | 5.275937336 |
 | 500 | 2.177081523| 3.479149128 | 4.173399578 |  5.229644753|
 | 1000| 1.886570943| 3.114268318| 3.564708046 | 4.133306292|




