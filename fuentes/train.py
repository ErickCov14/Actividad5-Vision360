from pathlib import Path
import argparse
import json

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def extraer_hog(imagen):
    return hog(
        imagen / 255.0,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )


def calcular_metricas(y_real, y_predicho, probabilidades):
    return {
        "accuracy": accuracy_score(y_real, y_predicho),
        "precision_macro": precision_score(
            y_real,
            y_predicho,
            average="macro",
            zero_division=0
        ),
        "recall_macro": recall_score(
            y_real,
            y_predicho,
            average="macro",
            zero_division=0
        ),
        "f1_macro": f1_score(
            y_real,
            y_predicho,
            average="macro",
            zero_division=0
        ),
        "roc_auc_ovr_weighted": roc_auc_score(
            y_real,
            probabilidades,
            multi_class="ovr",
            average="weighted"
        )
    }


def entrenar_modelo(nombre, pipeline, parametros, X_train, y_train, X_val, y_val, cv):
    busqueda = GridSearchCV(
        estimator=pipeline,
        param_grid=parametros,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1
    )

    with mlflow.start_run(run_name=f"{nombre}_GridSearch"):
        busqueda.fit(X_train, y_train)

        modelo = busqueda.best_estimator_
        prediccion = modelo.predict(X_val)
        probabilidades = modelo.predict_proba(X_val)
        metricas = calcular_metricas(y_val, prediccion, probabilidades)

        mlflow.log_param("modelo", nombre)
        mlflow.log_param("caracteristicas", "HOG")
        mlflow.log_param("cv_folds", 3)
        mlflow.log_params(
            {
                f"mejor_{clave.replace('modelo__', '')}": str(valor)
                for clave, valor in busqueda.best_params_.items()
            }
        )
        mlflow.log_metric("cv_mejor_f1_macro", busqueda.best_score_)

        for nombre_metrica, valor in metricas.items():
            mlflow.log_metric(f"validacion_{nombre_metrica}", valor)

        mlflow.log_dict(
            {
                "mejores_parametros": busqueda.best_params_,
                "mejor_f1_cv": busqueda.best_score_
            },
            f"grid_{nombre.lower().replace(' ', '_')}.json"
        )

        if nombre == "Random Forest":
            mlflow.sklearn.log_model(
                modelo,
                name="modelo",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
            )
        else:
            mlflow.sklearn.log_model(modelo, name="modelo")

    return modelo, busqueda, metricas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument(
        "--experiment",
        default="Actividad5_Vision360_RIAWELC"
    )
    args = parser.parse_args()

    ruta_salida = Path(args.output_dir)
    ruta_salida.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    datos = np.load(args.data, allow_pickle=True)
    X = datos["X"]
    y = datos["y"]
    particiones = datos["particiones"]

    X_hog = np.array([extraer_hog(imagen) for imagen in X])

    X_train = X_hog[particiones == "training"]
    y_train = y[particiones == "training"]

    X_val = X_hog[particiones == "validation"]
    y_val = y[particiones == "validation"]

    X_test = X_hog[particiones == "testing"]
    y_test = y[particiones == "testing"]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    modelos = {
        "Regresion Logistica": (
            Pipeline(
                [
                    ("escalado", StandardScaler()),
                    (
                        "modelo",
                        LogisticRegression(
                            max_iter=2000,
                            solver="lbfgs",
                            random_state=42
                        )
                    )
                ]
            ),
            {
                "modelo__C": [0.1, 1.0, 10.0],
                "modelo__class_weight": [None, "balanced"]
            }
        ),
        "Random Forest": (
            Pipeline(
                [
                    (
                        "modelo",
                        RandomForestClassifier(
                            random_state=42,
                            n_jobs=1
                        )
                    )
                ]
            ),
            {
                "modelo__n_estimators": [100, 200],
                "modelo__max_depth": [None, 20],
                "modelo__min_samples_split": [2, 5]
            }
        )
    }

    resultados = []
    modelos_entrenados = {}

    for nombre, (pipeline, parametros) in modelos.items():
        modelo, busqueda, metricas_val = entrenar_modelo(
            nombre,
            pipeline,
            parametros,
            X_train,
            y_train,
            X_val,
            y_val,
            cv
        )

        pred_prueba = modelo.predict(X_test)
        prob_prueba = modelo.predict_proba(X_test)
        metricas_prueba = calcular_metricas(y_test, pred_prueba, prob_prueba)

        resultados.append(
            {
                "modelo": nombre,
                "f1_cv_media": busqueda.best_score_,
                **{f"validacion_{k}": v for k, v in metricas_val.items()},
                **{f"prueba_{k}": v for k, v in metricas_prueba.items()}
            }
        )

        modelos_entrenados[nombre] = modelo

    comparacion = pd.DataFrame(resultados)
    comparacion.to_csv(
        ruta_salida / "comparacion_modelos.csv",
        index=False,
        encoding="utf-8-sig"
    )

    mejor_nombre = comparacion.sort_values(
        "validacion_f1_macro",
        ascending=False
    ).iloc[0]["modelo"]

    X_final = np.vstack([X_train, X_val])
    y_final = np.concatenate([y_train, y_val])

    modelo_final = modelos_entrenados[mejor_nombre]
    modelo_final.fit(X_final, y_final)

    joblib.dump(
        modelo_final,
        ruta_salida / "modelo_final.joblib"
    )

    with open(ruta_salida / "configuracion_final.json", "w", encoding="utf-8") as archivo:
        json.dump(
            {
                "modelo_seleccionado": mejor_nombre,
                "imagenes_entrenamiento_final": int(len(X_final))
            },
            archivo,
            ensure_ascii=False,
            indent=2
        )

    print(comparacion.round(4))
    print(f"Modelo seleccionado: {mejor_nombre}")


if __name__ == "__main__":
    main()
