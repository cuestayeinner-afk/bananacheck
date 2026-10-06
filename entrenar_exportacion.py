#!/usr/bin/env python3
"""Entrena un clasificador YOLOv8 con Exportacion/Rechazo."""

from pathlib import Path
import shutil

from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
DATASET = BASE / "dataset"
PREPARADO = BASE / "dataset_clasificacion"
MODELO_SALIDA = BASE / "modelo_exportacion.pt"


def preparar_dataset():
    """Crea la estructura train/val requerida por YOLO Classification."""
    for split, porcentaje in (("train", 0.8), ("val", 0.2)):
        for clase in ("Exportacion", "Rechazo"):
            destino = PREPARADO / split / clase
            destino.mkdir(parents=True, exist_ok=True)
            imagenes = sorted((DATASET / clase).glob("*"))
            corte = int(len(imagenes) * porcentaje)
            seleccion = imagenes[:corte] if split == "train" else imagenes[corte:]
            for imagen in seleccion:
                shutil.copy2(imagen, destino / imagen.name)


def main():
    if not (DATASET / "Exportacion").exists() or not (DATASET / "Rechazo").exists():
        raise SystemExit("Faltan dataset/Exportacion o dataset/Rechazo")
    preparar_dataset()
    modelo = YOLO("yolov8n-cls.pt")
    modelo.train(
        data=str(PREPARADO),
        epochs=30,
        imgsz=224,
        batch=16,
        device="cpu",
        project=str(BASE / "runs"),
        name="clasificador_exportacion",
        exist_ok=True,
    )
    mejor = BASE / "runs" / "classify" / "clasificador_exportacion" / "weights" / "best.pt"
    if mejor.exists():
        shutil.copy2(mejor, MODELO_SALIDA)
        print(f"Modelo guardado en: {MODELO_SALIDA}")
    else:
        print(f"No se encontró el modelo entrenado en: {mejor}")


if __name__ == "__main__":
    main()
