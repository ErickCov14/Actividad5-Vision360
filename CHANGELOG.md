# CHANGELOG - ACTIVIDAD 5 VISION 360

## Version 1.0 - 19 de octubre de 2026

### Datos

- Se utilizó el dataset público RIAWELC como muestra de radiografías de soldadura.
- Se conservaron las particiones originales de entrenamiento, validación y prueba.
- Se seleccionaron 2,000 imágenes de manera estratificada con semilla 42.
- Se procesaron las imágenes en escala de grises con tamaño de 64 × 64 píxeles.
- Se guardaron los datos limpios en formato NPZ y su manifiesto en CSV.
- Se incorporaron al repositorio las 2,000 imágenes originales de la muestra y un manifiesto con rutas relativas.

### Modelos

- Se extrajeron 324 características HOG por imagen.
- Se compararon Regresión Logística y Random Forest.
- Se realizó ajuste de hiperparámetros con Grid Search y validación cruzada estratificada de tres pliegues.
- Se seleccionó Random Forest mediante el F1 macro del conjunto de validación.
- Se entrenó un modelo final con las imágenes de entrenamiento y validación.

### Resultados

- F1 macro de validación para Random Forest: 0.6534.
- F1 macro de prueba para el modelo evaluado: 0.6627.
- Accuracy de prueba para el modelo evaluado: 0.6675.
- ROC-AUC de prueba para el modelo evaluado: 0.8858.
- Las métricas de prueba se documentan como referencia del modelo evaluado antes del reentrenamiento final.

### Trazabilidad

- Se registraron parámetros, métricas y modelos con MLflow.
- Se utilizó SQLite para almacenar los registros del experimento.
- Se actualizaron los resultados del repositorio con la ejecución más reciente.
- Se corrigió la selección del modelo en `train.py` para utilizar validación.

## Trabajo previo - 18 de octubre de 2026

- Se exploró el dataset y se identificaron sus cuatro clases.
- Se realizaron ejecuciones iniciales para probar la preparación de datos, los modelos y MLflow.
- Se ajustó la configuración de MLflow y se resolvió el problema de registro del modelo Random Forest.
- Los resultados de esas ejecuciones iniciales se sustituyeron en el repositorio por los de la ejecución más reciente.
