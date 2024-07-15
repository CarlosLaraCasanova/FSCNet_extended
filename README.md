# Estimación de localización y visibilidad de landmarks cefalométricos.

Repositorio con el código correspondiente a la parte informática del TFG "Análisis tensorial de redes
neuronales convolucionales y su aplicación a la estimación de la localización de landmarks cefalométricos 
y su visibilidad".

Por Carlos Lara Casanova

En `scripts` se encuentran los scripts utilizados para las ejecuciones realizadas para este TFG en los
servidores ngpu de la UGR. 

## Uso

En `dltrain/arguments.py` se encuentran los distintos argumentos que se pueden pasar al programa.

```bash
python entrenamiento.py --tosave ./path/to/save/results --cv /path/to/train_test_index.json --gt ./path/to/ground_truth.csv --boxes /path/to/precomputed_boxes.npy --ver /path/to/precomputed_vertices.npy --opt_mat /path/to/optimization_matrices.json --model name_of_model_to_train  --dataset name_of_dataset --label name_of_experiment
```

Para el entrenamiento del modelo híbrido se ejecuta `entrenamiento_hibrido` y para el modelo con normales `entrenamiento_normales`. Para este último hay que pasar adicionalmente la lista de normales precomputados. Las imágenes con las que se entrena/testea se asume que están en la carpeta `./FORENSE`.

Se asume que este código comparte ámbito con el código en [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2).

