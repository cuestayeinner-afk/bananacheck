#!/usr/bin/env python3
"""
BananaCheck AI — Servidor web con guardado de datos en CSV
Reemplaza: python3 -m http.server 9090
"""

import http.server
import json
import os
import csv
import hashlib
import hmac
import logging
from logging.handlers import RotatingFileHandler
import secrets
import time
from http.cookies import SimpleCookie
from io import BytesIO, StringIO
import bananacheck_storage as storage
try:
    import pandas as pd
    PANDAS_DISPONIBLE = True
except ImportError:
    PANDAS_DISPONIBLE = False
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs

# Importar YOLOv8 (opcional)
try:
    from yolo_inference import inicializar as init_yolo, inicializar_clasificador
    YOLO_DISPONIBLE = True
except (ImportError, SystemExit):
    YOLO_DISPONIBLE = False
    print("⚠️  yolo_inference no disponible — análisis sin YOLOv8")

# ── Configuración ──
PUERTO      = 9090
DIRECTORIO  = os.path.dirname(os.path.abspath(__file__))
CSV_FRUTA   = os.path.join(DIRECTORIO, "datos_fruta.csv")
CSV_TERRENO = os.path.join(DIRECTORIO, "datos_terreno.csv")
CSV_COMPARACIONES = os.path.join(DIRECTORIO, "analisis_guardados.csv")
CSV_FINCAS = os.path.join(DIRECTORIO, "fincas.csv")
CSV_HISTORIAL = os.path.join(DIRECTORIO, "historial_analisis.csv")
CLAVE_STAFF = os.environ.get("BANANACHECK_STAFF_PASSWORD") or "STAFF"
NOMBRE_COOKIE = "bananacheck_session"
DURACION_SESION = 8 * 60 * 60
SESIONES = {}
INTENTOS_LOGIN = {}
DATASET_SOIL = os.path.join(
    DIRECTORIO, "dataset", "soil.v1i.folder", "test"
)
ARCHIVO_LOG = os.path.join(storage.DATA_DIR, "bananacheck.log")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
logger = logging.getLogger("bananacheck")

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
    os.makedirs(DIRECTORIO, exist_ok=True)
    if not os.path.exists(CSV_FRUTA):
        with open(CSV_FRUTA, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(COLS_FRUTA)
        print(f"✅ CSV fruta creado: {CSV_FRUTA}")
    if not os.path.exists(CSV_TERRENO):
        with open(CSV_TERRENO, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(COLS_TERRENO)
        print(f"✅ CSV terreno creado: {CSV_TERRENO}")
    if not os.path.exists(CSV_COMPARACIONES):
        with open(CSV_COMPARACIONES, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["fecha_hora", "tipo", "salud_pct", "resumen"])
    if not os.path.exists(CSV_FINCAS):
        with open(CSV_FINCAS, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["fecha_registro", "nombre", "ubicacion", "registrada_por"])
    if not os.path.exists(CSV_HISTORIAL):
        with open(CSV_HISTORIAL, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["fecha_hora", "tipo", "usuario", "salud_pct", "resumen"])

def cargar_fincas():
    return storage.list_farms()

def registrar_finca(nombre, ubicacion, registrada_por):
    nombre = str(nombre or "").strip()
    ubicacion = str(ubicacion or "").strip()
    if not nombre or len(nombre) > 120:
        raise ValueError("El nombre de la finca es obligatorio y debe tener hasta 120 caracteres")
    if not ubicacion or len(ubicacion) > 180:
        raise ValueError("La ubicación es obligatoria y debe tener hasta 180 caracteres")
    try:
        return storage.add_farm(nombre, ubicacion, registrada_por)
    except Exception as error:
        if "UNIQUE constraint failed" in str(error):
            raise ValueError("Esa finca ya está registrada en esa ubicación") from error
        raise

def guardar_fruta(datos, usuario=None):
    """Completa los metadatos de un análisis de fruta antes de persistirlo."""
    datos["fecha_hora"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if usuario is not None:
        datos["usuario"] = usuario
    return storage.count_analyses("fruta") + 1

def guardar_terreno(datos, usuario=None):
    """Completa los metadatos de un análisis de terreno antes de persistirlo."""
    datos["fecha_hora"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if usuario is not None:
        datos["usuario"] = usuario
    return storage.count_analyses("terreno") + 1

def _texto_normalizado(valor):
    return str(valor or "").strip().lower()

def _es_desconocido(valor):
    texto = _texto_normalizado(valor)
    return not texto or any(palabra in texto for palabra in (
        "no concluyente", "no determinado", "no se puede", "desconocido", "sin datos"
    ))

def _defecto_presente(valor):
    texto = _texto_normalizado(valor)
    if _es_desconocido(texto):
        return None
    if any(palabra in texto for palabra in ("ninguno", "ninguna", "ningun", "sin ", "ausencia", "no hay", "no presenta", "no se observa", "no visible")):
        return False
    if texto in ("no", "no.", "falso") or texto.startswith("no "):
        return False
    return True

def _puntaje_componentes(componentes):
    evaluados = [(peso, sano) for peso, sano in componentes if sano is not None]
    if not evaluados:
        return None
    peso_total = sum(peso for peso, _ in evaluados)
    puntos = sum(peso for peso, sano in evaluados if sano)
    return round(100 * puntos / peso_total)

def puntaje_salud(tipo, datos):
    """Estima salud con indicadores visibles; omite valores no concluyentes."""
    if tipo == "fruta":
        return _puntaje_componentes([
            (40, not _defecto_presente(datos.get("signos_enfermedad")) if _defecto_presente(datos.get("signos_enfermedad")) is not None else None),
            (30, not _defecto_presente(datos.get("manchas_oscuras")) if _defecto_presente(datos.get("manchas_oscuras")) is not None else None),
            (30, not _defecto_presente(datos.get("golpes_deformaciones")) if _defecto_presente(datos.get("golpes_deformaciones")) is not None else None),
        ])

    humedad = _texto_normalizado(datos.get("humedad"))
    materia = _texto_normalizado(datos.get("materia_organica"))
    cobertura = _texto_normalizado(datos.get("cobertura_vegetal"))
    apto = _texto_normalizado(datos.get("apto_siembra"))
    humedad_sana = None if _es_desconocido(humedad) else (
        False if any(p in humedad for p in ("seco", "insuficiente", "encharcado", "excesiva"))
        else True if any(p in humedad for p in ("humedo", "húmedo", "adecuada", "suficiente"))
        else None
    )
    materia_sana = None if _es_desconocido(materia) else (
        False if any(p in materia for p in ("baja", "poca", "escasa", "deficiente"))
        else True if any(p in materia for p in ("presente", "alta", "abundante", "adecuada"))
        else None
    )
    cobertura_sana = None if _es_desconocido(cobertura) else (
        False if any(p in cobertura for p in ("falta", "ausente", "sin cobertura", "escasa"))
        else True if any(p in cobertura for p in ("adecuada", "presente", "buena", "abundante"))
        else None
    )
    apto_sano = None if _es_desconocido(apto) else (
        False if any(p in apto for p in ("no apto", "no", "inadecuado"))
        else True if any(p in apto for p in ("apto", "si", "sí", "adecuado"))
        else None
    )
    return _puntaje_componentes([
        (30, apto_sano),
        (25, None if _es_desconocido(datos.get("erosion")) else not _defecto_presente(datos.get("erosion"))),
        (25, None if _es_desconocido(datos.get("compactacion")) else not _defecto_presente(datos.get("compactacion"))),
        (10, humedad_sana),
        (5, materia_sana),
        (5, cobertura_sana),
    ])

def _cargar_comparaciones():
    return storage.list_comparisons()

def _cargar_historial(usuario=None):
    return storage.list_history(usuario)

def _hash_credencial(usuario, contrasena, rol):
    sal = secrets.token_bytes(16)
    digest = hashlib.scrypt(contrasena.encode("utf-8"), salt=sal, n=2**14, r=8, p=1)
    return {"usuario": usuario, "rol": rol, "sal": sal.hex(), "hash": digest.hex()}

def _cargar_cuentas():
    return storage.list_accounts()

def _guardar_cuentas(usuarios, crear=False):
    storage.save_accounts(usuarios, create=crear)

class BananaCheckHandler(http.server.SimpleHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORIO, **kwargs)

    def log_message(self, format, *args):
        logger.info("%s %s", self.address_string(), format % args)

    def _es_ruta_privada(self):
        ruta_solicitada = os.path.realpath(self.translate_path(urlparse(self.path).path))
        directorio_privado = os.path.realpath(storage.DATA_DIR)
        try:
            return os.path.commonpath((ruta_solicitada, directorio_privado)) == directorio_privado
        except ValueError:
            return False

    def do_GET(self):
        ruta = urlparse(self.path)
        if self._es_ruta_privada() or ruta.path in ("/bananacheck.db", "/bananacheck.log"):
            self.send_error(404)
            return
        if ruta.path == "/auth/session":
            self._auth_session()
            return
        if ruta.path == "/comparaciones":
            if not self._autenticado():
                self._json_response({"ok": False, "error": "Inicia sesión"}, 401)
                return
            try:
                self._json_response({"registros": _cargar_comparaciones()}, 200)
            except Exception as error:
                self._json_response({"error": str(error)}, 500)
            return
        if ruta.path == "/fincas":
            if not self._autenticado():
                self._json_response({"ok": False, "error": "Inicia sesión"}, 401)
                return
            self._json_response({"fincas": cargar_fincas()}, 200)
            return
        if ruta.path == "/mi-historial.csv":
            sesion = self._sesion_actual()
            if not sesion:
                self._json_response({"ok": False, "error": "Inicia sesión"}, 401)
                return
            if sesion["rol"] != "USUARIO":
                self._json_response({"ok": False, "error": "La exportación es exclusiva de usuarios"}, 403)
                return
            self._exportar_analisis_csv(parse_qs(ruta.query), usuario=sesion["usuario"])
            return
        if ruta.path == "/staff/model-metrics":
            if not self._sesion_staff():
                return
            self._metricas_modelo()
            return
        if ruta.path.startswith("/analisis/") and ruta.path.endswith("/pdf"):
            sesion = self._sesion_actual()
            if not sesion:
                self._json_response({"ok": False, "error": "Inicia sesión"}, 401)
                return
            try:
                analysis_id = int(ruta.path.split("/")[2])
            except (IndexError, ValueError):
                self._json_response({"ok": False, "error": "Análisis no encontrado"}, 404)
                return
            registro = storage.get_analysis(analysis_id)
            if not registro:
                self._json_response({"ok": False, "error": "Análisis no encontrado"}, 404)
                return
            if sesion["rol"] != "USUARIO" or registro["username"].casefold() != sesion["usuario"].casefold():
                self._json_response({"ok": False, "error": "La descarga PDF es exclusiva del propietario con rol de usuario"}, 403)
                return
            self._responder_pdf(registro)
            return
        if ruta.path == "/mi-historial":
            sesion = self._sesion_actual()
            if not sesion:
                self._json_response({"ok": False, "error": "Inicia sesión"}, 401)
                return
            try:
                self._json_response({"registros": _cargar_historial(sesion["usuario"])}, 200)
            except Exception as error:
                self._json_response({"error": str(error)}, 500)
            return
        if ruta.path == "/staff/dashboard":
            sesion = self._sesion_staff()
            if not sesion:
                return
            try:
                fincas = cargar_fincas()
                comparaciones = _cargar_comparaciones()
                historial = _cargar_historial()
                filtro = (parse_qs(ruta.query).get("tipo", ["todos"])[0] or "todos").lower()
                if filtro != "todos":
                    historial = [item for item in historial if item.get("tipo", "").casefold() == filtro.casefold()]
                puntajes = [
                    int(registro["salud_pct"])
                    for registro in comparaciones
                    if str(registro.get("salud_pct", "")).isdigit()
                ]
                tendencia = {"fruta": [], "terreno": []}
                for tipo in ("fruta", "terreno"):
                    for offset in range(6, -1, -1):
                        fecha = (datetime.now() - timedelta(days=offset)).strftime("%Y-%m-%d")
                        total = sum(1 for item in historial if item.get("tipo") == tipo and item.get("fecha_hora", "")[:10] == fecha)
                        tendencia[tipo].append({"fecha": fecha, "total": total})
                self._json_response({
                    "metricas": {
                        "fincas": len(fincas),
                        "analisis_fruta": storage.count_analyses("fruta"),
                        "analisis_terreno": storage.count_analyses("terreno"),
                        "comparaciones": len(comparaciones),
                        "promedio_salud_pct": round(sum(puntajes) / len(puntajes)) if puntajes else None,
                        "filtro_actual": filtro
                    },
                    "fincas": list(reversed(fincas[-10:])),
                    "historial": list(reversed(historial[-10:])),
                    "tendencia": tendencia,
                    "filtro": filtro
                }, 200)
            except Exception as error:
                self._json_response({"error": str(error)}, 500)
            return
        if ruta.path == "/staff/fincas":
            sesion = self._sesion_staff()
            if not sesion:
                return
            self._json_response({"fincas": list(reversed(cargar_fincas()))}, 200)
            return
        if ruta.path not in ("/", "/index.html") and not self._autenticado():
            self._json_response({"ok": False, "error": "Inicia sesión"}, 401)
            return
        if ruta.path == "/clasificar/suelo":
            self._clasificar_suelo_nombre(parse_qs(ruta.query).get("nombre", [""])[0])
            return
        super().do_GET()

    def do_HEAD(self):
        ruta = urlparse(self.path).path
        if self._es_ruta_privada() or ruta in ("/bananacheck.db", "/bananacheck.log"):
            self.send_error(404)
            return
        if ruta not in ("/", "/index.html") and not self._autenticado():
            self.send_error(401, "Inicia sesión")
            return
        super().do_HEAD()

    def do_OPTIONS(self):
        self.send_response(200)
        self._headers_cors()
        self.end_headers()

    def do_POST(self):
        ruta = urlparse(self.path).path

        if ruta == "/auth/session":
            self._auth_session()
        elif ruta == "/auth/setup":
            self._auth_setup()
        elif ruta == "/auth/login":
            self._auth_login()
        elif ruta == "/auth/logout":
            self._auth_logout()
        elif ruta == "/auth/change-password":
            self._auth_change_password()
        elif not self._autenticado():
            self._json_response({"ok": False, "error": "Inicia sesión"}, 401)
        elif ruta == "/staff/fincas":
            sesion = self._sesion_staff()
            if sesion:
                self._registrar_finca(sesion)
        elif ruta == "/guardar/fruta":
            self._guardar("/guardar/fruta", guardar_fruta, "fruta")
        elif ruta == "/guardar/terreno":
            self._guardar("/guardar/terreno", guardar_terreno, "terreno")
        elif ruta == "/stats":
            self._stats()
        elif ruta == "/yolo/detectar":
            self._yolo_detectar()
        elif ruta == "/clasificar/exportacion":
            self._clasificar_exportacion()
        elif ruta == "/clasificar/suelo":
            self._clasificar_suelo()
        else:
            self.send_response(404)
            self.end_headers()

    def _cuerpo_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if length > 1024 * 1024:
            raise ValueError("La solicitud supera el tamaño permitido")
        return json.loads(self.rfile.read(length) or b"{}")

    def _token_sesion(self):
        try:
            cookie = SimpleCookie(self.headers.get("Cookie", ""))
            morsel = cookie.get(NOMBRE_COOKIE)
            return morsel.value if morsel else ""
        except Exception:
            return ""

    def _autenticado(self):
        return self._sesion_actual() is not None

    def _sesion_staff(self):
        sesion = self._sesion_actual()
        if not sesion:
            self._json_response({"ok": False, "error": "Inicia sesión"}, 401)
            return None
        if sesion["rol"] not in ("STAFF", "ADMIN"):
            self._json_response({"ok": False, "error": "Acceso exclusivo de STAFF"}, 403)
            return None
        return sesion

    def _registrar_finca(self, sesion):
        try:
            datos = self._cuerpo_json()
            total = registrar_finca(datos.get("nombre"), datos.get("ubicacion"), sesion["usuario"])
            self._json_response({"ok": True, "total": total}, 201)
        except ValueError as error:
            self._json_response({"ok": False, "error": str(error)}, 400)
        except Exception as error:
            self._json_response({"ok": False, "error": str(error)}, 500)

    def _exportar_analisis_csv(self, parametros, usuario=None):
        filtros = {clave: valores[0] for clave, valores in parametros.items() if valores}
        if usuario is not None:
            filtros.pop("username", None)
            filtros["username"] = usuario
        tipo = filtros.get("tipo", "")
        if tipo not in ("", "todos", "fruta", "terreno"):
            self._json_response({"ok": False, "error": "Tipo de análisis no válido"}, 400)
            return
        for clave in ("desde", "hasta"):
            if filtros.get(clave):
                try:
                    datetime.strptime(filtros[clave], "%Y-%m-%d")
                except ValueError:
                    self._json_response({"ok": False, "error": "Las fechas deben tener formato AAAA-MM-DD"}, 400)
                    return
        if tipo == "todos":
            filtros.pop("tipo", None)
        registros = storage.list_analyses(filtros)
        columnas = ["id", "type", "created_at", "username", "farm", "health_score"]
        for registro in registros:
            for clave in registro:
                if clave not in columnas and clave not in ("fecha_hora", "usuario", "finca"):
                    columnas.append(clave)
        salida = StringIO(newline="")
        escritor = csv.DictWriter(salida, fieldnames=columnas, extrasaction="ignore")
        escritor.writeheader()
        for registro in registros:
            fila = {
                clave: (f"'{valor}" if isinstance(valor, str) and valor.lstrip(" \t\r").startswith(("=", "+", "-", "@")) else valor)
                for clave, valor in registro.items()
            }
            escritor.writerow(fila)
        contenido = salida.getvalue().encode("utf-8-sig")
        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", 'attachment; filename="bananacheck-analisis.csv"')
        self.send_header("Content-Length", str(len(contenido)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(contenido)

    def _metricas_modelo(self):
        metricas = []
        ruta_resultados = os.path.join(DIRECTORIO, "results.csv")
        if os.path.isfile(ruta_resultados):
            with open(ruta_resultados, "r", newline="", encoding="utf-8-sig") as source:
                filas = list(csv.DictReader(source))
            campos = (
                ("metrics/precision(B)", "Precisión de detección"),
                ("metrics/recall(B)", "Recall de detección"),
                ("metrics/mAP50(B)", "mAP50 de detección"),
                ("metrics/mAP50-95(B)", "mAP50-95 de detección"),
                ("metrics/precision(M)", "Precisión de segmentación"),
                ("metrics/recall(M)", "Recall de segmentación"),
                ("metrics/mAP50(M)", "mAP50 de segmentación"),
                ("metrics/mAP50-95(M)", "mAP50-95 de segmentación"),
            )
            for campo, etiqueta in campos:
                validas = [fila for fila in filas if fila.get(campo)]
                if validas:
                    mejor = max(validas, key=lambda fila: float(fila[campo] or 0))
                    metricas.append({"label": etiqueta, "value": float(mejor[campo]), "epoch": mejor.get("epoch", "")})
        self._json_response({"metrics": metricas}, 200)

    def _responder_pdf(self, registro):
        try:
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.graphics.shapes import Drawing, Rect, String
            from xml.sax.saxutils import escape

            contenido = BytesIO()
            documento = SimpleDocTemplate(contenido, pagesize=letter, rightMargin=0.65 * inch, leftMargin=0.65 * inch)
            estilos = getSampleStyleSheet()
            estilos.add(ParagraphStyle(name="BananaTitle", parent=estilos["Title"], textColor=colors.HexColor("#215b43"), alignment=TA_CENTER, spaceAfter=14))
            estilos.add(ParagraphStyle(name="FieldValue", parent=estilos["BodyText"], wordWrap="CJK"))
            historia = [Paragraph("BananaCheck · Informe de análisis", estilos["BananaTitle"])]
            tipo = registro["type"]
            datos = registro["payload"]
            etiquetas = {
                "fruta": [("color_cascara", "Color de cáscara"), ("etapa_unece", "Etapa UNECE"), ("textura", "Textura"), ("manchas_oscuras", "Manchas oscuras"), ("golpes_deformaciones", "Golpes o deformaciones"), ("signos_enfermedad", "Signos de enfermedad"), ("apto_nacional", "Apto nacional"), ("apto_internacional", "Apto internacional"), ("diagnostico", "Diagnóstico")],
                "terreno": [("color_suelo", "Color del suelo"), ("humedad", "Humedad"), ("cobertura_vegetal", "Cobertura vegetal"), ("erosion", "Erosión"), ("compactacion", "Compactación"), ("materia_organica", "Materia orgánica"), ("apto_siembra", "Apto para siembra"), ("problemas", "Problemas"), ("recomendacion", "Recomendación")],
            }
            puntaje = registro.get("health_score")
            datos_base = [
                ["Tipo", "Fruta" if tipo == "fruta" else "Terreno"],
                ["Fecha", registro.get("created_at", "")],
                ["Usuario", registro.get("username") or ""],
                ["Finca", registro.get("farm") or "Sin especificar"],
                ["Salud estimada", f"{puntaje}%" if puntaje is not None else "Sin datos concluyentes"],
            ]
            tabla_base = Table(datos_base, colWidths=[1.45 * inch, 5.4 * inch])
            tabla_base.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#edf4ef")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#215b43")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d6e0d8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]))
            historia.extend([tabla_base, Spacer(1, 14)])
            dibujo = Drawing(450, 36)
            dibujo.add(String(0, 22, "Indicador visual de salud", fontSize=9, fillColor=colors.HexColor("#46554b")))
            dibujo.add(Rect(0, 3, 440, 12, fillColor=colors.HexColor("#e5ece7"), strokeColor=None))
            if puntaje is not None:
                color = colors.HexColor("#bc5b35") if int(puntaje) < 50 else colors.HexColor("#d5a329") if int(puntaje) < 75 else colors.HexColor("#36805c")
                dibujo.add(Rect(0, 3, 440 * max(0, min(100, int(puntaje))) / 100, 12, fillColor=color, strokeColor=None))
            historia.extend([dibujo, Spacer(1, 10), Paragraph("Resultados", estilos["Heading2"])])
            filas = [["Indicador", "Resultado"]]
            for clave, etiqueta in etiquetas[tipo]:
                valor = str(datos.get(clave) or "Sin dato")
                filas.append([etiqueta, Paragraph(escape(valor).replace("\n", "<br/>"), estilos["FieldValue"])])
            tabla = Table(filas, colWidths=[1.8 * inch, 5.05 * inch], repeatRows=1)
            tabla.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#215b43")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d6e0d8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8f6")]),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]))
            historia.append(tabla)
            documento.build(historia)
            pdf = contenido.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Disposition", f'attachment; filename="bananacheck-{tipo}-{registro["id"]}.pdf"')
            self.send_header("Content-Length", str(len(pdf)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(pdf)
        except ImportError:
            self._json_response({"ok": False, "error": "Falta instalar ReportLab para generar PDF"}, 503)

    def _sesion_actual(self):
        token = self._token_sesion()
        sesion = SESIONES.get(token)
        if not sesion or sesion["caduca"] <= time.time():
            SESIONES.pop(token, None)
            return None
        return sesion

    def _auth_session(self):
        sesion = self._sesion_actual()
        self._json_response({
            "autenticado": sesion is not None,
            "usuario": sesion["usuario"] if sesion else None,
            "rol": sesion["rol"] if sesion else None,
            "configurar_cuenta": not storage.has_accounts()
        }, 200)

    def _auth_setup(self):
        if self.client_address[0] not in ("127.0.0.1", "::1"):
            self._json_response({"ok": False, "error": "Configura la cuenta desde este equipo"}, 403)
            return
        if storage.has_accounts():
            self._json_response({"ok": False, "error": "La cuenta ya está configurada"}, 409)
            return
        try:
            datos = self._cuerpo_json()
            usuario = str(datos.get("usuario", "")).strip()
            contrasena = str(datos.get("contrasena", ""))
            if len(usuario) < 3 or len(usuario) > 64:
                raise ValueError("El usuario debe tener entre 3 y 64 caracteres")
            if len(contrasena) < 10:
                raise ValueError("La contraseña debe tener al menos 10 caracteres")
            _guardar_cuentas([_hash_credencial(usuario, contrasena, "ADMIN")], crear=True)
            self._iniciar_sesion(usuario, "ADMIN")
        except FileExistsError:
            self._json_response({"ok": False, "error": "La cuenta ya está configurada"}, 409)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._json_response({"ok": False, "error": str(error)}, 400)
        except Exception as error:
            self._json_response({"ok": False, "error": str(error)}, 500)

    def _login_esta_limitado(self):
        ahora = time.monotonic()
        ip = self.client_address[0]
        for direccion, momentos in tuple(INTENTOS_LOGIN.items()):
            recientes = [momento for momento in momentos if ahora - momento < 300]
            if recientes:
                INTENTOS_LOGIN[direccion] = recientes
            else:
                INTENTOS_LOGIN.pop(direccion, None)
        recientes = INTENTOS_LOGIN.get(ip, [])
        if len(recientes) >= 5:
            self._json_response({"ok": False, "error": "Demasiados intentos. Espera cinco minutos."}, 429)
            return True
        return False

    def _login_fallido(self):
        ip = self.client_address[0]
        intentos = INTENTOS_LOGIN.setdefault(ip, [])
        intentos.append(time.monotonic())
        estado = 429 if len(intentos) >= 5 else 401
        mensaje = "Demasiados intentos. Espera cinco minutos." if estado == 429 else "Usuario o contraseña incorrectos"
        logger.warning("Fallo de autenticación desde %s; estado=%s", ip, estado)
        self._json_response({"ok": False, "error": mensaje}, estado)

    def _auth_login(self):
        if self._login_esta_limitado():
            return
        if not storage.has_accounts():
            self._json_response({"ok": False, "configurar_cuenta": True}, 409)
            return
        try:
            datos = self._cuerpo_json()
            usuario = str(datos.get("usuario", "")).strip()
            contrasena = str(datos.get("contrasena", ""))

            if hmac.compare_digest(contrasena.encode("utf-8"), CLAVE_STAFF.encode("utf-8")):
                INTENTOS_LOGIN.pop(self.client_address[0], None)
                self._iniciar_sesion(usuario or "STAFF", "STAFF")
                return

            cuentas = _cargar_cuentas()
            cuenta = next(
                (item for item in cuentas if item.get("usuario", "").casefold() == usuario.casefold()),
                None
            )

            if cuenta:
                sal = bytes.fromhex(cuenta["sal"])
                hash_esperado = bytes.fromhex(cuenta["hash"])
                hash_ingresado = hashlib.scrypt(
                    contrasena.encode("utf-8"),
                    salt=sal, n=2**14, r=8, p=1
                )
                if hmac.compare_digest(hash_ingresado, hash_esperado):
                    INTENTOS_LOGIN.pop(self.client_address[0], None)
                    self._iniciar_sesion(cuenta["usuario"], cuenta.get("rol", "ADMIN"))
                    return
                self._login_fallido()
                return

            if usuario and len(usuario) >= 3 and len(contrasena) >= 5:
                cuentas.append(_hash_credencial(usuario, contrasena, "USUARIO"))
                _guardar_cuentas(cuentas)
                INTENTOS_LOGIN.pop(self.client_address[0], None)
                self._iniciar_sesion(usuario, "USUARIO")
                return

            self._login_fallido()
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            self._json_response({"ok": False, "error": "No se pudo validar la cuenta"}, 400)

    def _iniciar_sesion(self, usuario, rol):
        token = secrets.token_urlsafe(32)
        SESIONES[token] = {
            "caduca": time.time() + DURACION_SESION,
            "usuario": usuario,
            "rol": rol
        }
        self._json_response(
            {"ok": True, "autenticado": True, "usuario": usuario, "rol": rol},
            200,
            cookie=f"{NOMBRE_COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age={DURACION_SESION}"
        )

    def _auth_logout(self):
        SESIONES.pop(self._token_sesion(), None)
        self._json_response(
            {"ok": True}, 200,
            cookie=f"{NOMBRE_COOKIE}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0"
        )

    def _auth_change_password(self):
        sesion = self._sesion_actual()
        if not sesion:
            self._json_response({"ok": False, "error": "Inicia sesión"}, 401)
            return
        try:
            datos = self._cuerpo_json()
            actual = str(datos.get("actual", ""))
            nueva = str(datos.get("nueva", ""))
            usuario = sesion["usuario"]
            if usuario == "STAFF":
                self._json_response({"ok": False, "error": "La clave STAFF no se puede cambiar desde aquí"}, 400)
                return
            if len(nueva) < 5:
                raise ValueError("La nueva contraseña debe tener al menos 5 caracteres")
            cuentas = _cargar_cuentas()
            cuenta = next((item for item in cuentas if item.get("usuario", "").casefold() == usuario.casefold()), None)
            if not cuenta:
                raise ValueError("Usuario no encontrado")
            sal = bytes.fromhex(cuenta["sal"])
            hash_esperado = bytes.fromhex(cuenta["hash"])
            hash_ingresado = hashlib.scrypt(actual.encode("utf-8"), salt=sal, n=2**14, r=8, p=1)
            if not hmac.compare_digest(hash_ingresado, hash_esperado):
                self._json_response({"ok": False, "error": "La contraseña actual no coincide"}, 401)
                return
            cuenta["sal"] = secrets.token_bytes(16).hex()
            cuenta["hash"] = hashlib.scrypt(nueva.encode("utf-8"), salt=bytes.fromhex(cuenta["sal"]), n=2**14, r=8, p=1).hex()
            _guardar_cuentas(cuentas)
            self._json_response({"ok": True, "mensaje": "Contraseña actualizada"}, 200)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            self._json_response({"ok": False, "error": str(error)}, 400)
        except Exception as error:
            self._json_response({"ok": False, "error": str(error)}, 500)

    def _guardar(self, ruta, fn_guardar, tipo):
        try:
            datos = self._cuerpo_json()
            usuario = self._sesion_actual()["usuario"] if self._sesion_actual() else None
            fn_guardar(datos, usuario=usuario)
            puntaje = puntaje_salud(tipo, datos)
            resumen = datos.get("diagnostico") or datos.get("recomendacion") or ""
            analysis_id, total = storage.save_analysis(tipo, datos, usuario, puntaje, resumen)
            self.send_response(200)
            self._headers_cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            resp = json.dumps({
                "ok": True,
                "id": analysis_id,
                "total": total,
                "tipo": tipo,
                "salud_pct": puntaje,
                "pdf_url": f"/analisis/{analysis_id}/pdf",
            })
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
            registros_fruta = storage.list_analyses({"type": "fruta"})
            registros_terreno = storage.list_analyses({"type": "terreno"})
            stats = {
                "fruta": {
                    "total": len(registros_fruta),
                    "aptos_nacional": sum(1 for row in registros_fruta if (row.get("apto_nacional") or "").lower() in ("sí", "si")),
                    "aptos_internacional": sum(1 for row in registros_fruta if (row.get("apto_internacional") or "").lower() in ("sí", "si")),
                    "ultimo": registros_fruta[0].get("fecha_hora") if registros_fruta else None,
                },
                "terreno": {
                    "total": len(registros_terreno),
                    "aptos_siembra": sum(1 for row in registros_terreno if any(kw in (row.get("apto_siembra") or "").lower() for kw in ("sí", "si", "apto"))),
                    "ultimo": registros_terreno[0].get("fecha_hora") if registros_terreno else None,
                },
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

    def _clasificar_suelo(self):
        """Consulta si una imagen pertenece a Soil o Not_Soil del dataset de test."""
        try:
            datos = self._cuerpo_json()
            self._clasificar_suelo_nombre(datos.get("nombre", ""))
        except Exception as e:
            self.send_response(500)
            self._headers_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"encontrada": False, "error": str(e)}).encode())

    def _clasificar_suelo_nombre(self, nombre):
        nombre = os.path.basename(str(nombre))
        try:
            clase = None
            for nombre_clase, etiqueta in (("Soil", "tierra"), ("Not_Soil", "no_tierra")):
                carpeta = os.path.join(DATASET_SOIL, nombre_clase)
                if os.path.isfile(os.path.join(carpeta, nombre)):
                    clase = etiqueta
                    break

            self.send_response(200)
            self._headers_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"encontrada": clase is not None, "clase": clase}).encode())
        except Exception as e:
            self.send_response(500)
            self._headers_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"encontrada": False, "error": str(e)}).encode())

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
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > MAX_UPLOAD_BYTES:
                self._json_response({"ok": False, "error": "La imagen debe pesar como máximo 10 MB"}, 413)
                return
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

    def _clasificar_exportacion(self):
        """Clasifica el banano como Exportacion o Rechazo."""
        if not YOLO_DISPONIBLE:
            self._json_response({"success": False, "error": "ultralytics no instalado"}, 503)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > MAX_UPLOAD_BYTES:
                self._json_response({"success": False, "error": "La imagen debe pesar como máximo 10 MB"}, 413)
                return
            imagen = self.rfile.read(length)
            resultado = inicializar_clasificador().clasificar(imagen)
            self._json_response(resultado, 200 if resultado["success"] else 503)
        except Exception as error:
            self._json_response({"success": False, "error": str(error)}, 500)

    def _json_response(self, datos, estado, cookie=None):
        self.send_response(estado)
        self._headers_cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(json.dumps(datos).encode())

    def _headers_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, ngrok-skip-browser-warning")

def create_server(host="0.0.0.0", port=PUERTO, handler_cls=BananaCheckHandler):
    try:
        http.server.HTTPServer.allow_reuse_address = True
        return http.server.ThreadingHTTPServer((host, port), handler_cls)
    except OSError as exc:
        if exc.errno in {98, 48}:
            raise RuntimeError(
                f"El puerto {port} ya está ocupado. Cierra la otra instancia de BananaCheck o cambia BANANACHECK_PORT/PUERTO antes de arrancar."
            ) from exc
        raise


if __name__ == "__main__":
    os.makedirs(storage.DATA_DIR, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.addHandler(RotatingFileHandler(ARCHIVO_LOG, maxBytes=1_000_000, backupCount=3, encoding="utf-8"))
    logger.handlers[0].setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    storage.initialize()
    inicializar_csv()
    if YOLO_DISPONIBLE:
        try:
            init_yolo()
            print("✅ YOLOv8 inicializado")
        except Exception as e:
            print(f"⚠️  Error inicializando YOLOv8: {e}")
    try:
        server = create_server()
    except RuntimeError as exc:
        print(f"❌ {exc}")
        raise SystemExit(1)
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
