from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm.auto import tqdm


MAPEO_CLASES = {
    "Difetto1": "Grieta",
    "Difetto2": "Porosidad",
    "Difetto4": "Falta de penetracion",
    "NoDifetto": "Sin defecto"
}

CLASE_A_ID = {
    "Grieta": 0,
    "Porosidad": 1,
    "Falta de penetracion": 2,
    "Sin defecto": 3
}

TAMANOS_MUESTRA = {
    "training": 300,
    "validation": 100,
    "testing": 100
}


def construir_inventario(ruta_dataset):
    registros = []

    for particion in TAMANOS_MUESTRA:
        for carpeta, clase in MAPEO_CLASES.items():
            ruta_clase = ruta_dataset / particion / carpeta

            for imagen in ruta_clase.glob("*.png"):
                registros.append(
                    {
                        "archivo": imagen.name,
                        "ruta": str(imagen),
                        "particion": particion,
                        "clase": clase
                    }
                )

    return pd.DataFrame(registros)


def seleccionar_muestra(df_inventario):
    muestras = []

    for particion, cantidad in TAMANOS_MUESTRA.items():
        for clase in CLASE_A_ID:
            grupo = df_inventario[
                (df_inventario["particion"] == particion)
                & (df_inventario["clase"] == clase)
            ]

            muestras.append(grupo.sample(n=cantidad, random_state=42))

    return pd.concat(muestras, ignore_index=True)


def limpiar_imagenes(df_muestra):
    imagenes = []
    etiquetas = []
    particiones = []
    archivos = []
    errores = []

    for _, fila in tqdm(
        df_muestra.iterrows(),
        total=len(df_muestra),
        desc="Limpiando imagenes"
    ):
        try:
            with Image.open(fila["ruta"]) as imagen:
                imagen = ImageOps.exif_transpose(imagen)
                imagen = imagen.convert("L")
                imagen = imagen.resize((64, 64), Image.Resampling.LANCZOS)

                imagenes.append(np.asarray(imagen, dtype=np.uint8))
                etiquetas.append(CLASE_A_ID[fila["clase"]])
                particiones.append(fila["particion"])
                archivos.append(fila["archivo"])

        except Exception as error:
            errores.append(
                {
                    "ruta": fila["ruta"],
                    "error": str(error)
                }
            )

    return (
        np.stack(imagenes),
        np.array(etiquetas),
        np.array(particiones),
        np.array(archivos),
        pd.DataFrame(errores)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    ruta_dataset = Path(args.input_dir)
    ruta_salida = Path(args.output_dir)
    ruta_salida.mkdir(parents=True, exist_ok=True)

    inventario = construir_inventario(ruta_dataset)
    muestra = seleccionar_muestra(inventario)

    X, y, particiones, archivos, errores = limpiar_imagenes(muestra)

    np.savez_compressed(
        ruta_salida / "riawelc_muestra_limpia_64x64.npz",
        X=X,
        y=y,
        particiones=particiones,
        archivos=archivos
    )

    muestra.to_csv(
        ruta_salida / "muestra_estratificada.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        {
            "archivo": archivos,
            "etiqueta": y,
            "particion": particiones
        }
    ).to_csv(
        ruta_salida / "manifest_muestra_limpia.csv",
        index=False,
        encoding="utf-8-sig"
    )

    errores.to_csv(
        ruta_salida / "errores_limpieza.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Imagenes limpias: {len(X)}")
    print(f"Imagenes con error: {len(errores)}")


if __name__ == "__main__":
    main()
