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
import secrets
import time
from http.cookies import SimpleCookie
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
ARCHIVO_CUENTA = os.path.join(DIRECTORIO, ".bananacheck_cuenta.json")
CLAVE_STAFF = os.environ.get("BANANACHECK_STAFF_PASSWORD") or "STAFF"
NOMBRE_COOKIE = "bananacheck_session"
DURACION_SESION = 8 * 60 * 60
SESIONES = {}
DATASET_SOIL = os.path.join(
    DIRECTORIO, "dataset", "soil.v1i.folder", "test"
)

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
    with open(CSV_FINCAS, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def registrar_finca(nombre, ubicacion, registrada_por):
    nombre = str(nombre or "").strip()
    ubicacion = str(ubicacion or "").strip()
    if not nombre or len(nombre) > 120:
        raise ValueError("El nombre de la finca es obligatorio y debe tener hasta 120 caracteres")
    if not ubicacion or len(ubicacion) > 180:
        raise ValueError("La ubicación es obligatoria y debe tener hasta 180 caracteres")
    existentes = cargar_fincas()
    duplicada = any(
        finca["nombre"].casefold() == nombre.casefold()
        and finca["ubicacion"].casefold() == ubicacion.casefold()
        for finca in existentes
    )
    if duplicada:
        raise ValueError("Esa finca ya está registrada en esa ubicación")
    with open(CSV_FINCAS, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), nombre, ubicacion, registrada_por
        ])
    return len(existentes) + 1

def _contar_registros(ruta):
    if not os.path.exists(ruta):
        return 0
    with open(ruta, "r", newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))

def guardar_fruta(datos, usuario=None):
    """Guarda un análisis de fruta en el CSV."""
    datos["fecha_hora"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if usuario is not None:
        datos["usuario"] = usuario
    nueva_fila = [datos.get(col, "") for col in COLS_FRUTA]
    with open(CSV_FRUTA, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(nueva_fila)
    with open(CSV_FRUTA, "r", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)

def guardar_terreno(datos, usuario=None):
    """Guarda un análisis de terreno en el CSV."""
    datos["fecha_hora"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if usuario is not None:
        datos["usuario"] = usuario
    nueva_fila = [datos.get(col, "") for col in COLS_TERRENO]
    with open(CSV_TERRENO, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(nueva_fila)
    with open(CSV_TERRENO, "r", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)

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

def _guardar_comparacion(tipo, datos):
    puntaje = puntaje_salud(tipo, datos)
    resumen = datos.get("diagnostico") if tipo == "fruta" else datos.get("recomendacion")
    if puntaje is None:
        resumen = f"{resumen or ''} (sin indicadores suficientes para puntuar)".strip()
    with open(CSV_COMPARACIONES, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), tipo,
            "" if puntaje is None else puntaje, resumen or ""
        ])
    return puntaje

def _cargar_comparaciones():
    with open(CSV_COMPARACIONES, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def _guardar_historial(tipo, usuario, salud_pct, resumen):
    if not usuario:
        return
    with open(CSV_HISTORIAL, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            str(tipo),
            str(usuario),
            "" if salud_pct is None else str(salud_pct),
            str(resumen or "")
        ])

def _cargar_historial(usuario=None):
    if not os.path.exists(CSV_HISTORIAL):
        return []
    with open(CSV_HISTORIAL, "r", newline="", encoding="utf-8") as f:
        registros = list(csv.DictReader(f))
    if usuario:
        registros = [r for r in registros if (r.get("usuario") or "").casefold() == usuario.casefold()]
    return registros

def _hash_credencial(usuario, contrasena, rol):
    sal = secrets.token_bytes(16)
    digest = hashlib.scrypt(contrasena.encode("utf-8"), salt=sal, n=2**14, r=8, p=1)
    return {"usuario": usuario, "rol": rol, "sal": sal.hex(), "hash": digest.hex()}

def _cargar_cuentas():
    with open(ARCHIVO_CUENTA, "r", encoding="utf-8") as archivo:
        datos = json.load(archivo)
    if isinstance(datos.get("usuarios"), list):
        return datos["usuarios"]
    if datos.get("usuario"):
        datos.setdefault("rol", "ADMIN")
        return [datos]
    raise ValueError("Formato de cuentas no válido")

def _guardar_cuentas(usuarios, crear=False):
    modo = "x" if crear else "w"
    ruta_temporal = ARCHIVO_CUENTA if crear else f"{ARCHIVO_CUENTA}.tmp"
    with open(ruta_temporal, modo, encoding="utf-8") as archivo:
        json.dump({"usuarios": usuarios}, archivo)
    os.chmod(ruta_temporal, 0o600)
    if not crear:
        os.replace(ruta_temporal, ARCHIVO_CUENTA)

class BananaCheckHandler(http.server.SimpleHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORIO, **kwargs)

    def log_message(self, format, *args):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.address_string()} — {format % args}")

    def do_GET(self):
        ruta = urlparse(self.path)
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
                        "analisis_fruta": sum(1 for item in historial if item.get("tipo") == "fruta") or _contar_registros(CSV_FRUTA),
                        "analisis_terreno": sum(1 for item in historial if item.get("tipo") == "terreno") or _contar_registros(CSV_TERRENO),
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
        if urlparse(self.path).path not in ("/", "/index.html") and not self._autenticado():
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
            "configurar_cuenta": not os.path.exists(ARCHIVO_CUENTA)
        }, 200)

    def _auth_setup(self):
        if self.client_address[0] not in ("127.0.0.1", "::1"):
            self._json_response({"ok": False, "error": "Configura la cuenta desde este equipo"}, 403)
            return
        if os.path.exists(ARCHIVO_CUENTA):
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

    def _auth_login(self):
        if not os.path.exists(ARCHIVO_CUENTA):
            self._json_response({"ok": False, "configurar_cuenta": True}, 409)
            return
        try:
            datos = self._cuerpo_json()
            usuario = str(datos.get("usuario", "")).strip()
            contrasena = str(datos.get("contrasena", ""))

            if hmac.compare_digest(contrasena.encode("utf-8"), CLAVE_STAFF.encode("utf-8")):
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
                    self._iniciar_sesion(cuenta["usuario"], cuenta.get("rol", "ADMIN"))
                    return
                self._json_response({"ok": False, "error": "Usuario o contraseña incorrectos"}, 401)
                return

            if usuario and len(usuario) >= 3 and len(contrasena) >= 5:
                cuentas.append(_hash_credencial(usuario, contrasena, "USUARIO"))
                _guardar_cuentas(cuentas)
                self._iniciar_sesion(usuario, "USUARIO")
                return

            self._json_response({"ok": False, "error": "Usuario o contraseña incorrectos"}, 401)
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
            total = fn_guardar(datos, usuario=usuario)
            puntaje = _guardar_comparacion(tipo, datos)
            _guardar_historial(tipo, usuario, puntaje, datos.get("diagnostico") or datos.get("recomendacion") or "")
            self.send_response(200)
            self._headers_cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            resp = json.dumps({"ok": True, "total": total, "tipo": tipo, "salud_pct": puntaje})
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
                with open(CSV_FRUTA, "r", encoding="utf-8") as f:
                    reader = list(csv.DictReader(f))
                    stats["fruta"] = {
                        "total": len(reader),
                        "aptos_nacional": sum(1 for r in reader if (r.get("apto_nacional") or "").lower() in ["sí", "si"]),
                        "aptos_internacional": sum(1 for r in reader if (r.get("apto_internacional") or "").lower() == "no"),
                        "ultimo": reader[-1]["fecha_hora"] if reader else None
                    }
            if os.path.exists(CSV_TERRENO):
                with open(CSV_TERRENO, "r", encoding="utf-8") as f:
                    reader = list(csv.DictReader(f))
                    stats["terreno"] = {
                        "total": len(reader),
                        "aptos_siembra": sum(1 for r in reader if any(kw in (r.get("apto_siembra") or "").lower() for kw in ["sí", "si", "apto"])),
                        "ultimo": reader[-1]["fecha_hora"] if reader else None
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
            length = int(self.headers.get("Content-Length", 0))
            datos = json.loads(self.rfile.read(length))
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

if __name__ == "__main__":
    inicializar_csv()
    if YOLO_DISPONIBLE:
        try:
            init_yolo()
            print("✅ YOLOv8 inicializado")
        except Exception as e:
            print(f"⚠️  Error inicializando YOLOv8: {e}")
    http.server.HTTPServer.allow_reuse_address = True
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
