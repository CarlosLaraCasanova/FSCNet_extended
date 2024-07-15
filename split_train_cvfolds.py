import pandas as pd
import random
import numpy as np
import os
import json

nombre_archivo_leer = "split_train_test.json"
nombre_archivo_guadar = "split_train_cvfolds.json"


f = open(nombre_archivo_leer)
datos = json.load(f)


idx_train = datos[0]

salida = []

divisiones = [26,26,26,27,27]


assert sum(divisiones)==len(idx_train)

idx_div_test = []

i = 0
for div in divisiones:
	idx_div_test.append([idx_train[j] for j in range(i,i+div)])
	i+= div


idx_div_train = []
for idx in idx_div_test:
	lista = idx_train.copy()
	for j in idx:
		lista.remove(j)
	idx_div_train.append(lista.copy())


for i in range(5):
	salida.append( [idx_div_train[i],idx_div_test[i]])
 
with open(nombre_archivo_guadar, "w") as f:
	f.write(str(salida))
