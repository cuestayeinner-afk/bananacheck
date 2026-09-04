#!/usr/bin/env python3
"""
YOLOv8 Inference para BananaCheck
Detecta y segmenta bananos en imágenes
"""

import os
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("⚠️  ultralytics no instalado. Instala con: pip install ultralytics")
    sys.exit(1)

import cv2
import numpy as np
from PIL import Image
import io

class BananoDetector:
    def __init__(self, model_path=None):
        """
        Inicializa detector YOLOv8 para bananos.
        
        Args:
            model_path: ruta a modelo .pt entrenado. Si es None, usa yolov8n-seg.pt
        """
        try:
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
                print(f"✅ Modelo cargado: {model_path}")
            else:
                # Descarga modelo base si no existe uno entrenado
                self.model = YOLO("yolov8n-seg.pt")
                print("✅ Modelo base YOLOv8n-seg descargado")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            self.model = None

    def detectar_bananos(self, image_path, conf_threshold=0.5):
        """
        Detecta bananos en imagen.
        
        Args:
            image_path: ruta o bytes de imagen
            conf_threshold: confianza mínima (0-1)
        
        Returns:
            dict con resultados
        """
        if self.model is None:
            return {
                "success": False,
                "error": "Modelo no cargado",
                "banano_detectado": False,
                "confianza": 0
            }

        try:
            # Cargar imagen
            if isinstance(image_path, str):
                img = cv2.imread(image_path)
            else:
                # Es bytes (de formulario web)
                img_array = np.frombuffer(image_path, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                return {
                    "success": False,
                    "error": "No se pudo cargar la imagen",
                    "banano_detectado": False,
                    "confianza": 0
                }

            # Inferencia
            results = self.model(img, conf=conf_threshold, verbose=False)
            
            if not results or len(results) == 0:
                return {
                    "success": True,
                    "error": None,
                    "banano_detectado": False,
                    "confianza": 0,
                    "detecciones": 0,
                    "diagnostico": "No se detectaron objetos"
                }

            result = results[0]
            boxes = result.boxes
            
            if len(boxes) == 0:
                return {
                    "success": True,
                    "error": None,
                    "banano_detectado": False,
                    "confianza": 0,
                    "detecciones": 0,
                    "diagnostico": "No se detectaron bananos"
                }

            # Analizar detecciones
            max_conf = float(boxes.conf.max().cpu().numpy()) if len(boxes.conf) > 0 else 0
            num_detecciones = len(boxes)
            
            # Criterios banano-valid: conf > threshold
            banano_valido = max_conf >= conf_threshold

            return {
                "success": True,
                "error": None,
                "banano_detectado": banano_valido,
                "confianza": float(max_conf),
                "detecciones": num_detecciones,
                "diagnostico": "Banano detectado ✓" if banano_valido else "No es un banano válido ✗"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "banano_detectado": False,
                "confianza": 0
            }

    def procesar_archivo(self, file_obj, conf_threshold=0.5):
        """
        Procesa archivo subido (desde formulario HTTP).
        
        Args:
            file_obj: objeto archivo con .read()
            conf_threshold: confianza mínima
        
        Returns:
            dict con resultados
        """
        try:
            image_bytes = file_obj.read()
            return self.detectar_bananos(image_bytes, conf_threshold)
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "banano_detectado": False,
                "confianza": 0
            }


# Instancia global
detector = None

def inicializar():
    """Inicializa detector al arrancar servidor."""
    global detector
    if detector is None:
        # Buscar modelo entrenado; si no existe, usa base
        modelo_entrenado = os.path.expanduser("~/Descargas/bananacheck/modelo_banano.pt")
        detector = BananoDetector(modelo_entrenado if os.path.exists(modelo_entrenado) else None)
    return detector


if __name__ == "__main__":
    # Test: python3 yolo_inference.py <imagen>
    if len(sys.argv) > 1:
        det = BananoDetector()
        resultado = det.detectar_bananos(sys.argv[1])
        print(resultado)
    else:
        print("Uso: python3 yolo_inference.py <ruta_imagen>")
