# ACTIVIDAD 5 - VISION 360

## CLASIFICACION DE DEFECTOS DE SOLDADURA CON MLFLOW

Este proyecto aplica preparación de datos, entrenamiento, ajuste de hiperparámetros y registro de experimentos a un caso de inspección de soldadura relacionado con Vision 360.

Las imágenes reales de producción y los registros operativos de Vision 360 no se incluyen por razones de confidencialidad. Para esta actividad se utiliza RIAWELC, un conjunto público de imágenes radiográficas de soldadura. Los resultados corresponden a una validación técnica inicial y no demuestran el desempeño con imágenes reales de producción.

## DATASET

- Nombre: RIAWELC
- Fuente: https://github.com/stefyste/RIAWELC
- Total de imágenes originales identificadas: 24,407
- Tipo de problema: clasificación multiclase
- Formato utilizado: PNG

| Carpeta original | Clase | Identificador |
| --- | --- | ---: |
| Difetto1 | Grieta | 0 |
| Difetto2 | Porosidad | 1 |
| Difetto4 | Falta de penetración | 2 |
| NoDifetto | Sin defecto | 3 |

Se seleccionó una muestra estratificada con semilla 42, conservando las particiones originales del dataset.

| Partición | Imágenes por clase | Total |
| --- | ---: | ---: |
| Entrenamiento | 300 | 1,200 |
| Validación | 100 | 400 |
| Prueba | 100 | 400 |
| Total | 500 | 2,000 |

## PREPARACION DE DATOS

1. Se identifican las imágenes PNG de las cuatro clases.
2. Se selecciona una muestra por clase y partición.
3. Se corrige la orientación de la imagen cuando los metadatos lo indican.
4. Las imágenes se convierten a escala de grises.
5. Se redimensionan a 64 × 64 píxeles.
6. Se genera un archivo NPZ y un manifiesto CSV.
7. Se extraen 324 características HOG por imagen.

En la ejecución reportada se procesaron 2,000 imágenes sin errores de lectura.

## MODELOS Y AJUSTE

Se comparan dos modelos:

- Regresión Logística con escalado de características
- Random Forest

Los hiperparámetros se ajustan con Grid Search y validación cruzada estratificada de tres pliegues. Se utiliza F1 macro para comparar los modelos y se selecciona el mejor con el conjunto de validación.

Las métricas son accuracy, precision macro, recall macro, F1 macro y ROC-AUC multiclase con estrategia one-vs-rest ponderada.

## RESULTADOS

| Modelo | F1 CV media | F1 validación | F1 prueba | Accuracy prueba | ROC-AUC prueba |
| --- | ---: | ---: | ---: | ---: | ---: |
| Regresión Logística | 0.5519 | 0.5650 | 0.5611 | 0.5700 | 0.8028 |
| Random Forest | 0.6567 | 0.6534 | 0.6627 | 0.6675 | 0.8858 |

Random Forest se seleccionó por obtener el mayor F1 macro en validación. Las métricas de prueba corresponden al modelo entrenado inicialmente con 1,200 imágenes.

Después se entrenó un modelo final con las 1,600 imágenes de entrenamiento y validación. Las métricas de prueba anteriores se conservan como referencia y no constituyen una evaluación directa del modelo reentrenado.

La desviación del F1 en los tres pliegues fue 0.0178 para Regresión Logística y 0.0109 para Random Forest. Esta comparación es descriptiva y no constituye una prueba de significancia estadística.

## MLFLOW

MLflow registra parámetros, métricas, artefactos y modelos. La ejecución en Colab utiliza una base SQLite dentro de la carpeta de trabajo del proyecto. Las métricas de prueba del notebook se registran en corridas de evaluación separadas.

## ESTRUCTURA

```text
Actividad5/
├── datos/
│   ├── datos_ini/
│   └── datos_limp/
├── fuentes/
│   ├── entrena.ipynb
│   ├── datos_prep.py
│   └── train.py
├── resultados/
├── evidencias_mlflow/
├── README.md
└── CHANGELOG.md
```

La carpeta `datos/datos_ini/muestra_original` contiene las 2,000 imágenes originales seleccionadas, organizadas por partición y clase. Su manifiesto incluye rutas relativas. El dataset completo se obtiene desde la fuente pública indicada.

La carpeta `datos/datos_limp` contiene la muestra procesada y su manifiesto. Las versiones de los paquetes instalados en el entorno actual se encuentran en `requirements.txt`.

## REPRODUCCION EN GOOGLE COLAB

1. Obtener RIAWELC desde su repositorio público.
2. Colocar el dataset en Google Drive.
3. Abrir `fuentes/entrena.ipynb` en Google Colab.
4. Seleccionar CPU y ejecutar las celdas en orden.
5. Autorizar el montaje de Google Drive.
6. Consultar las tablas, gráficas y registros de MLflow.

El notebook utiliza la ruta `/content/drive/MyDrive/Actividad5_Vision360` y espera encontrar el dataset en `datos/datos_ini/RIAWELC-main/Dataset_partitioned/DB - Copy`. Si se utiliza otra ubicación, se deben ajustar las rutas de la primera celda.

## REPRODUCCION MEDIANTE SCRIPTS

Desde la carpeta `Actividad5`, instalar las dependencias:

```bash
python -m pip install -r requirements.txt
```

Preparar los datos. Sustituir `/ruta/al/dataset` por la carpeta que contiene `training`, `validation` y `testing`:

```bash
python fuentes/datos_prep.py --input-dir "/ruta/al/dataset" --output-dir "datos/datos_limp"
```

Entrenar y registrar los experimentos:

```bash
python fuentes/train.py --data "datos/datos_limp/riawelc_muestra_limpia_64x64.npz" --output-dir "resultados" --tracking-uri "sqlite:///mlflow.db"
```

El script genera una comparación y un modelo final. Las métricas de validación y los modelos ajustados se registran en MLflow. Algunos nombres de salida son diferentes a los del notebook.

## LIMITACIONES Y TRABAJO FUTURO

La muestra es pequeña frente al dataset completo y no representa necesariamente las condiciones de producción de Vision 360. Antes de considerar un despliegue se requiere validar con imágenes reales autorizadas, revisar las etiquetas con personal experto y analizar los errores por clase.

También se debe evaluar la independencia entre imágenes relacionadas, documentar las versiones del entorno, monitorear cambios en los datos y mantener revisión humana de las predicciones.
