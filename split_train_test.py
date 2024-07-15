import pandas as pd
import random
import numpy as np
import os

nombre_archivo_guardar = "split_train_test.json"

df = pd.read_csv("./FORENSE/data_curated_bbox.csv")
filenames = df.iloc[:, 0]
np_dataset_puntos_list = df.iloc[:, 1:].to_numpy().reshape((-1, 30, 2))
np_dataset_visibility_list = ~np.isnan(np_dataset_puntos_list)[:,:,0]



a = [i for i in range(len(filenames))]
random.Random(4610).shuffle(a)

n_train = int(0.8*len(a))
n_test = int(0.2*len(a))

idx_train = [a[i] for i in range(n_train)]
idx_test = [a[i] for i in range(n_train,len(a))]


visibles_train = np.zeros(30) 
visibles_test = np.zeros(30)


for i in idx_train:
	for lm in range(30):
		visibles_train[lm] += int(np_dataset_visibility_list[i][lm])

for i in idx_test:
	for lm in range(30):
		visibles_test[lm] += int(np_dataset_visibility_list[i][lm])


maximo = max(abs(visibles_train/len(idx_train) - visibles_test/len(idx_test)))

salida = [idx_train,idx_test]

with open(nombre_archivo_guardar, "w") as f:
    f.write(str(salida))
