#!/usr/bin/env python3
"""Entrena un clasificador visual con las categorías comerciales revisadas."""

from itertools import combinations
from pathlib import Path
import random
import re
import shutil

BASE = Path(__file__).resolve().parent
DATASET = BASE / "dataset_exportacion_clasificado"
DATASET_BINARIO = BASE / "dataset"
DIRECTORIO_PREPARADO = BASE / "dataset_clasificacion"
MODELO_SALIDA = BASE / "modelo_exportacion.pt"
CLASES_TRES = ("Internacional", "Nacional", "Rechazo")
CLASES_BINARIAS = ("Exportacion", "Rechazo")
CLASES = CLASES_TRES
SEMILLA = 42
FRACCION_VALIDACION = 0.2
EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PATRON_SESION = re.compile(r"IMG_(\d{8})", re.IGNORECASE)
PATRON_LOTE = re.compile(r"^(LOTE|RACIMO|SESION|SESSION)[_-]([A-Z0-9-]+)", re.IGNORECASE)
MARCADOR = ".bananacheck_split"


def _obtener_grupo(imagen):
    lote = PATRON_LOTE.match(imagen.stem)
    if lote:
        return f"{lote.group(1).upper()}:{lote.group(2).upper()}"
    coincidencia = PATRON_SESION.search(imagen.stem)
    if not coincidencia:
        raise ValueError(
            f"No se puede inferir la sesión de captura de {imagen.name}. "
            "Usa nombres IMG_YYYYMMDD... o prefijos LOTE-<id>_, "
            "RACIMO-<id>_ o SESION-<id>_ para mantener grupos juntos."
        )
    return coincidencia.group(1)


def _particionar_grupos(imagenes_por_clase, semilla=SEMILLA, clases=CLASES_TRES):
    """Separa sesiones completas y busca proporciones equilibradas por clase."""
    grupos_por_clase = {
        clase: {} for clase in clases
    }
    grupos = {}
    for clase, imagenes in imagenes_por_clase.items():
        for imagen in imagenes:
            grupo = _obtener_grupo(imagen)
            grupos.setdefault(grupo, []).append((clase, imagen))
            grupos_por_clase[clase].setdefault(grupo, 0)
            grupos_por_clase[clase][grupo] += 1

    for clase, grupos_clase in grupos_por_clase.items():
        if len(grupos_clase) < 2:
            raise ValueError(
                f"La clase {clase} necesita al menos dos sesiones de captura "
                "distintas para separar entrenamiento y validación."
            )

    rng = random.Random(semilla)
    ids_grupos = sorted(grupos)
    rng.shuffle(ids_grupos)
    total_por_clase = {
        clase: sum(grupos_por_clase[clase].values()) for clase in clases
    }
    candidatos = []
    if len(ids_grupos) <= 18:
        candidatos = (
            set(subconjunto)
            for cantidad in range(1, len(ids_grupos))
            for subconjunto in combinations(ids_grupos, cantidad)
        )
    else:
        candidatos = (
            set(rng.sample(ids_grupos, rng.randint(1, len(ids_grupos) - 1)))
            for _ in range(10000)
        )

    mejor_validacion = None
    mejor_puntaje = float("inf")
    for validacion in candidatos:
        conteos = {clase: 0 for clase in clases}
        for grupo in validacion:
            for clase, _ in grupos[grupo]:
                conteos[clase] += 1
        if any(conteos[clase] == 0 or conteos[clase] == total_por_clase[clase]
               for clase in clases):
            continue
        puntaje = sum(
            abs(conteos[clase] / total_por_clase[clase] - FRACCION_VALIDACION)
            for clase in clases
        )
        if puntaje < mejor_puntaje:
            mejor_puntaje = puntaje
            mejor_validacion = validacion

    if mejor_validacion is None:
        raise ValueError(
            "No se pudo separar sesiones sin mezclar clases entre entrenamiento "
            "y validación. Añade más sesiones de captura etiquetadas."
        )

    particiones = {"train": [], "val": []}
    for grupo, elementos in grupos.items():
        split = "val" if grupo in mejor_validacion else "train"
        particiones[split].extend(elementos)
    for split in particiones:
        particiones[split].sort(key=lambda item: (item[0], item[1].name))
    return particiones


def _imagenes_de_clases(directorio, clases):
    carpetas = [directorio / clase for clase in clases]
    existentes = [carpeta.is_dir() for carpeta in carpetas]
    if not any(existentes):
        return None
    if not all(existentes):
        faltantes = [clase for clase, existe in zip(clases, existentes) if not existe]
        raise ValueError(
            f"Conjunto incompleto en {directorio}; faltan carpetas: "
            f"{', '.join(faltantes)}."
        )

    imagenes_por_clase = {}
    for clase in clases:
        carpeta = directorio / clase
        imagenes = sorted(
            imagen for imagen in carpeta.iterdir()
            if imagen.is_file() and imagen.suffix.lower() in EXTENSIONES_IMAGEN
        ) if carpeta.is_dir() else []
        if not imagenes:
            raise ValueError(f"No hay imágenes etiquetadas en {carpeta}.")
        imagenes_por_clase[clase] = imagenes
    return imagenes_por_clase


def preparar_dataset(semilla=SEMILLA):
    """Prepara categorías de mercado o la clasificación general disponible."""
    imagenes_por_clase = _imagenes_de_clases(DATASET, CLASES_TRES)
    if imagenes_por_clase:
        clases = CLASES_TRES
        nombre = "tres_categorias"
        print("Entrenamiento con etiquetas de mercado revisadas.")
    else:
        imagenes_por_clase = _imagenes_de_clases(DATASET_BINARIO, CLASES_BINARIAS)
        if not imagenes_por_clase:
            raise ValueError(
                "Faltan las carpetas dataset_exportacion_clasificado/"
                "{Internacional,Nacional,Rechazo} y también las carpetas "
                "dataset/{Exportacion,Rechazo}."
            )
        clases = CLASES_BINARIAS
        nombre = "exportacion_binaria"
        print(
            "AVISO: se entrenará solo Exportacion/Rechazo. "
            "Este dataset no permite distinguir mercado nacional e internacional."
        )

    particiones = _particionar_grupos(imagenes_por_clase, semilla, clases)
    preparado = DIRECTORIO_PREPARADO / nombre
    preparado.mkdir(parents=True, exist_ok=True)
    marcador = preparado / MARCADOR
    if marcador.exists():
        for split in ("train", "val"):
            split_dir = preparado / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
    elif any(preparado.iterdir()):
        raise FileExistsError(
            f"{preparado} ya contiene datos sin el marcador esperado; "
            "no se sobrescribirá."
        )
    marcador.write_text("Generado por entrenar_exportacion.py\n", encoding="utf-8")

    for split, elementos in particiones.items():
        for clase, imagen in elementos:
            destino = preparado / split / clase
            destino.mkdir(parents=True, exist_ok=True)
            shutil.copy2(imagen, destino / imagen.name)

    return preparado, nombre


def main():
    try:
        preparado, modo = preparar_dataset()
    except (FileNotFoundError, ValueError, FileExistsError) as error:
        raise SystemExit(str(error)) from error

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit(
            "Falta ultralytics. Instala las dependencias YOLO antes de entrenar."
        ) from error

    modelo = YOLO("yolov8n-cls.pt")
    modelo.train(
        data=str(preparado),
        epochs=20,
        imgsz=224,
        batch=16,
        device="cpu",
        project=str(BASE / "runs"),
        name=f"clasificador_{modo}",
        exist_ok=True,
        seed=SEMILLA,
        workers=0,
    )
    mejor = Path(modelo.trainer.best)
    if mejor.exists():
        shutil.copy2(mejor, MODELO_SALIDA)
        print(f"Modelo guardado en: {MODELO_SALIDA}")
    else:
        raise SystemExit(f"No se encontró el modelo entrenado en: {mejor}")


if __name__ == "__main__":
    main()
