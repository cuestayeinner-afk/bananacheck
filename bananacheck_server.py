#!/usr/bin/env python3
"""
BananaCheck AI — Servidor web con guardado de datos en CSV
Reemplaza: python3 -m http.server 9090
"""

import http.server
import json
import os
import pandas as pd
from datetime import datetime
from urllib.parse import urlparse, parse_qs

# Importar YOLOv8 (opcional)
try:
    from yolo_inference import inicializar as init_yolo, detector
    YOLO_DISPONIBLE = True
except ImportError:
    YOLO_DISPONIBLE = False
    print("⚠️  yolo_inference no disponible — análisis sin YOLOv8")

# ── Configuración ──
PUERTO      = 9090
DIRECTORIO  = os.path.expanduser("~/Descargas/bananacheck")
CSV_FRUTA   = os.path.expanduser("~/Descargas/bananacheck/datos_fruta.csv")
CSV_TERRENO = os.path.expanduser("~/Descargas/bananacheck/datos_terreno.csv")

# ── Columnas del dataframe de fruta ──
COLS_FRUTA = [
    "fecha_hora",
    "color_cascara",
    "etapa_unece",
    "textura",
    "manchas_oscuras",
    "golpes_deformaciones",
    "signos_enfermedad",
    "apto_nacional",
    "apto_internacional",
    "diagnostico"
]

# ── Columnas del dataframe de terreno ──
COLS_TERRENO = [
    "fecha_hora",
    "color_suelo",
    "humedad",
    "cobertura_vegetal",
    "erosion",
    "compactacion",
    "materia_organica",
    "apto_siembra",
    "problemas",
    "recomendacion"
]

def inicializar_csv():
    """Crea los CSV si no existen."""
    if not os.path.exists(CSV_FRUTA):
        pd.DataFrame(columns=COLS_FRUTA).to_csv(CSV_FRUTA, index=False)
        print(f"✅ CSV fruta creado: {CSV_FRUTA}")
    if not os.path.exists(CSV_TERRENO):
        pd.DataFrame(columns=COLS_TERRENO).to_csv(CSV_TERRENO, index=False)
        print(f"✅ CSV terreno creado: {CSV_TERRENO}")

def guardar_fruta(datos):
    """Guarda un análisis de fruta en el CSV."""
    datos["fecha_hora"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.read_csv(CSV_FRUTA)
    nueva_fila = {col: datos.get(col, "") for col in COLS_FRUTA}
    df = pd.concat([df, pd.DataFrame([nueva_fila])], ignore_index=True)
    df.to_csv(CSV_FRUTA, index=False)
    return len(df)

def guardar_terreno(datos):
    """Guarda un análisis de terreno en el CSV."""
    datos["fecha_hora"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.read_csv(CSV_TERRENO)
    nueva_fila = {col: datos.get(col, "") for col in COLS_TERRENO}
    df = pd.concat([df, pd.DataFrame([nueva_fila])], ignore_index=True)
    df.to_csv(CSV_TERRENO, index=False)
    return len(df)

class BananaCheckHandler(http.server.SimpleHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORIO, **kwargs)

    def log_message(self, format, *args):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.address_string()} — {format % args}")

    def do_OPTIONS(self):
        self.send_response(200)
        self._headers_cors()
        self.end_headers()

    def do_POST(self):
        ruta = urlparse(self.path).path

        if ruta == "/guardar/fruta":
            self._guardar("/guardar/fruta", guardar_fruta, "fruta")
        elif ruta == "/guardar/terreno":
            self._guardar("/guardar/terreno", guardar_terreno, "terreno")
        elif ruta == "/stats":
            self._stats()
        elif ruta == "/yolo/detectar":
            self._yolo_detectar()
        else:
            self.send_response(404)
            self.end_headers()

    def _guardar(self, ruta, fn_guardar, tipo):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            datos  = json.loads(body)
            total  = fn_guardar(datos)
            self.send_response(200)
            self._headers_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = json.dumps({"ok": True, "total": total, "tipo": tipo})
            self.wfile.write(resp.encode())
            print(f"💾 Guardado análisis de {tipo} — total registros: {total}")
        except Exception as e:
            self.send_response(500)
            self._headers_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode())
            print(f"❌ Error guardando {tipo}: {e}")

    def _stats(self):
        try:
            stats = {}
            if os.path.exists(CSV_FRUTA):
                df = pd.read_csv(CSV_FRUTA)
                stats["fruta"] = {
                    "total": len(df),
                    "aptos_nacional": int((df["apto_nacional"].str.lower() == "sí").sum()),
                    "aptos_internacional": int((df["apto_internacional"].str.lower() == "no").sum()),
                    "ultimo": df["fecha_hora"].iloc[-1] if len(df) > 0 else None
                }
            if os.path.exists(CSV_TERRENO):
                df = pd.read_csv(CSV_TERRENO)
                stats["terreno"] = {
                    "total": len(df),
                    "aptos_siembra": int((df["apto_siembra"].str.lower().str.contains("sí|si|apto", na=False)).sum()),
                    "ultimo": df["fecha_hora"].iloc[-1] if len(df) > 0 else None
                }
            self.send_response(200)
            self._headers_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(stats).encode())
        except Exception as e:
            self.send_response(500)
            self._headers_cors()
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _yolo_detectar(self):
        """Endpoint para detección YOLOv8 de bananos."""
        if not YOLO_DISPONIBLE:
            self.send_response(503)
            self._headers_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "ok": False,
                "error": "YOLOv8 no disponible",
                "banano_detectado": False
            }).encode())
            return

        try:
            # Leer imagen multipart
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" in content_type:
                # Parsear multipart
                import cgi
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        'REQUEST_METHOD': 'POST',
                        'CONTENT_TYPE': content_type,
                    }
                )
                if "imagen" not in form:
                    raise ValueError("Campo 'imagen' no encontrado")
                fileitem = form["imagen"]
                image_data = fileitem.file.read()
            else:
                # Raw bytes
                length = int(self.headers.get("Content-Length", 0))
                image_data = self.rfile.read(length)

            if not image_data:
                raise ValueError("Imagen vacía")

            # Detección
            det = init_yolo() if YOLO_DISPONIBLE else None
            if det is None:
                raise RuntimeError("Detector no inicializado")
            
            resultado = det.detectar_bananos(image_data, conf_threshold=0.5)

            self.send_response(200)
            self._headers_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resultado).encode())
            print(f"🎯 YOLOv8: {resultado}")

        except Exception as e:
            self.send_response(500)
            self._headers_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "ok": False,
                "error": str(e),
                "banano_detectado": False
            }).encode())
            print(f"❌ Error YOLOv8: {e}")

    def _headers_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, ngrok-skip-browser-warning")

if __name__ == "__main__":
    inicializar_csv()
    if YOLO_DISPONIBLE:
        try:
            init_yolo()
            print("✅ YOLOv8 inicializado")
        except Exception as e:
            print(f"⚠️  Error inicializando YOLOv8: {e}")
    server = http.server.HTTPServer(("0.0.0.0", PUERTO), BananaCheckHandler)
    print(f"""
╔══════════════════════════════════════╗
║   BananaCheck AI — Servidor activo   ║
╠══════════════════════════════════════╣
║  Puerto:  {PUERTO}                        ║
║  Carpeta: ~/Descargas/bananacheck    ║
║  CSV:     datos_fruta.csv            ║
║           datos_terreno.csv          ║
╚══════════════════════════════════════╝
""")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Servidor detenido.")
