## ***********************************************
## Importamos librerias y módulos a utilizar
## ***********************************************
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from utils import loadDS

## ***********************************************
## Cargamos set de datos
## ***********************************************
path = os.path.join("datasets", "housing")

housing = loadDS(path, "housing.csv")

## ***********************************************
## Rápida mirada de nuestro set de datos
## ***********************************************

# Imprimimos las primeras 5 filas
housing.head() # Cada fila representa un distrito.

# Veamos los atributos (columnas) que conforman mi set de datos.
print(housing.columns)

## El método info() nos da una rápidad descripción del tipo de datos que tenemos en el dataset
print(housing.info())

"""
There are 20,640 instances in the dataset, which means that it is fairly small by
Machine Learning standards, but it’s perfect to get started. Notice that the
total_bedrooms attribute has only 20,433 nonnull values, meaning that 207
districts are missing this feature. We will need to take care of this later

All attributes are numerical, except the ocean_proximity field. Its type is
object, so it could hold any kind of Python object

All attributes are numerical, except the ocean_proximity field. Its type is
object, so it could hold any kind of Python object. But since you loaded this
data from a CSV file, you know that it must be a text attribute. When you
looked at the top five rows, you probably noticed that the values in the
ocean_proximity column were repetitive, which means that it is probably a
categorical attribute. You can find out what categories exist and how many
districts belong to each category by using the value_counts() method:
"""

housing["ocean_proximity"].value_counts()

## Resumen de nuestros datos numéricos
housing.describe()

## Resumen de sólo alguno de los datos 
housing[["housing_median_age","total_rooms","total_bedrooms","population","households"]].describe()

## Graficando histogramas
plt.style.use("seaborn")
housing.hist(bins=50, figsize=(20,15))
plt.show()

import numpy as np
import matplotlib.pyplot as plt

## Generamos datos 
rng = np.random.RandomState(10)
X = rng.randn(200, 2) #posición de los puntos en el espacio
y = np.dot(X, [-2, 1]) + 2 * rng.randn(X.shape[0]) #etiquetas de los puntos

datos = np.hstack([X, y[:, None]])
datos.shape

plt.style.use("seaborn")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')

ax.scatter(datos[:,0], datos[:,1], datos[:,2],c=y, s=40, cmap='winter')
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Labels")
ax.set_title("Datos de entrada")

# ax.view_init(-10, -120)
plt.show()