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
    YOLO = None

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

class BananoDetector:
    def __init__(self, model_path=None):
        """
        Inicializa detector YOLOv8 para bananos.
        
        Args:
            model_path: ruta a modelo .pt entrenado. Si es None, usa yolov8n-seg.pt
        """
        if YOLO is None or cv2 is None or np is None:
            self.model = None
            return
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
        if self.model is None or cv2 is None or np is None:
            return {
                "success": False,
                "error": "Modelo no cargado",
                "banano_detectado": False,
                "confianza": 0
            }


class ExportacionClassifier:
    """Clasifica fruta y se abstiene cuando la confianza es insuficiente."""

    UMBRAL_CONFIANZA = 0.70

    def __init__(self, model_path=None):
        self.model = None
        if YOLO is not None and model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)

    @classmethod
    def decidir_categoria(cls, nombre, confianza):
        """Aplica la relación de categorías y la regla de abstención."""
        confianza = float(confianza)
        categoria = str(nombre).strip().casefold()
        if confianza < cls.UMBRAL_CONFIANZA:
            return {
                "clasificacion": "No concluyente",
                "nivel_exportacion": None,
                "apto_exportacion": "No concluyente",
                "apto_nacional": "No concluyente",
                "apto_internacional": "No concluyente",
                "confianza": confianza,
            }

        if categoria == "internacional":
            nivel = "Internacional"
            nacional, internacional = "Sí", "Sí"
        elif categoria == "nacional":
            nivel = "Nacional"
            nacional, internacional = "Sí", "No"
        elif categoria == "rechazo":
            nivel = "Rechazo"
            nacional, internacional = "No", "No"
        elif categoria == "exportacion":
            return {
                "clasificacion": "Exportación general (mercado no especificado)",
                "nivel_exportacion": "General",
                "apto_exportacion": "Sí",
                "apto_nacional": "Sí",
                "apto_internacional": "Sí",
                "confianza": confianza,
            }
        else:
            raise ValueError(
                f"Categoría de modelo no reconocida: {nombre!r}. "
                "Se esperaba Internacional, Nacional, Exportacion o Rechazo."
            )
        return {
            "clasificacion": nivel,
            "nivel_exportacion": nivel,
            "apto_exportacion": "No" if nivel == "Rechazo" else "Sí",
            "apto_nacional": nacional,
            "apto_internacional": internacional,
            "confianza": confianza,
        }

    def clasificar(self, image_bytes):
        if self.model is None:
            return {
                "success": False,
                "error": (
                    "Clasificador no cargado. Ejecuta entrenar_exportacion.py "
                    "para entrenar con las etiquetas disponibles."
                ),
                "clasificacion": None,
                "confianza": 0,
            }
        try:
            array = np.frombuffer(image_bytes, np.uint8)
            imagen = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if imagen is None:
                raise ValueError("No se pudo cargar la imagen")
            resultado = self.model(imagen, verbose=False)[0]
            indice = int(resultado.probs.top1)
            confianza = float(resultado.probs.top1conf)
            nombre = str(resultado.names[indice])
            decision = self.decidir_categoria(nombre, confianza)
            return {"success": True, **decision}
        except Exception as error:
            return {"success": False, "error": str(error), "clasificacion": None, "confianza": 0}

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
clasificador_exportacion = None

def inicializar():
    """Inicializa detector al arrancar servidor."""
    global detector
    if detector is None:
        # Buscar modelo entrenado; si no existe, usa base
        modelo_entrenado = os.path.expanduser("~/Descargas/bananacheck/modelo_banano.pt")
        detector = BananoDetector(modelo_entrenado if os.path.exists(modelo_entrenado) else None)
    return detector


def inicializar_clasificador():
    global clasificador_exportacion
    if clasificador_exportacion is None:
        modelo = Path(__file__).resolve().parent / "modelo_exportacion.pt"
        clasificador_exportacion = ExportacionClassifier(modelo)
    return clasificador_exportacion


if __name__ == "__main__":
    # Test: python3 yolo_inference.py <imagen>
    if len(sys.argv) > 1:
        det = BananoDetector()
        resultado = det.detectar_bananos(sys.argv[1])
        print(resultado)
    else:
        print("Uso: python3 yolo_inference.py <ruta_imagen>")
