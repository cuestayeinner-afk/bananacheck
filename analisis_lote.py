#!/usr/bin/env python3
"""
BananaCheck AI — Análisis en lote de imágenes de banano
Procesa todas las fotos del dataset y guarda resultados en CSV
"""

import os
import json
import base64
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

# ── Configuración ──
DATASET_DIR  = os.path.expanduser("~/Descargas/bananacheck/dataset")
CSV_SALIDA   = os.path.expanduser("~/Descargas/bananacheck/dataset_analizado.csv")
OLLAMA_URL   = "http://127.0.0.1:11434"
MODELO       = "llava"
EXTENSIONES  = {".jpg", ".jpeg", ".png", ".webp"}

# ── Columnas del dataframe ──
COLUMNAS = [
    "archivo",
    "fecha_hora",
    "color_cascara",
    "etapa_unece",
    "textura",
    "manchas_oscuras",
    "golpes_deformaciones",
    "signos_enfermedad",
    "apto_nacional",
    "apto_internacional",
    "diagnostico",
    "respuesta_completa"
]

PROMPT = """Analiza esta imagen de banano y responde SOLO con esta tabla en español:

| Campo | Valor |
|-------|-------|
| Color de la cáscara | |
| Etapa UNECE (1-7) | |
| Textura visible | |
| Manchas oscuras | |
| Golpes o deformaciones | |
| Signos de enfermedad | |
| Apto exportación nacional (sí/no) | |
| Apto exportación internacional (sí/no) | |
| Diagnóstico breve | |

Responde SOLO la tabla, en español, sin texto adicional."""

def imagen_a_base64(ruta, max_px=512):
    """Convierte imagen a base64 redimensionada."""
    try:
        from PIL import Image
        import io
        img = Image.open(ruta)
        img.thumbnail((max_px, max_px), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        # Sin PIL, usar base64 directo
        with open(ruta, "rb") as f:
            return base64.b64encode(f.read()).decode()

def analizar_imagen(ruta_imagen):
    """Llama a Ollama con la imagen y devuelve la respuesta."""
    b64 = imagen_a_base64(ruta_imagen)
    payload = {
        "model": MODELO,
        "messages": [
            {
                "role": "system",
                "content": "Eres un experto en calidad de banano. Respondes SIEMPRE en español."
            },
            {
                "role": "user",
                "content": PROMPT,
                "images": [b64]
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 300,
            "num_ctx": 512
        }
    }
    res = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=180)
    res.raise_for_status()
    return res.json()["message"]["content"].strip()

def extraer_valor(texto, campo):
    """Extrae valor de una tabla markdown."""
    import re
    patron = re.compile(campo + r'[^|\n]*\|\s*([^|\n]+)', re.IGNORECASE)
    m = patron.search(texto.lower())
    return m.group(1).strip() if m else ""

def parsear_respuesta(texto, archivo):
    """Convierte la respuesta del modelo en un diccionario."""
    return {
        "archivo":               os.path.basename(archivo),
        "fecha_hora":            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "color_cascara":         extraer_valor(texto, "color de la c"),
        "etapa_unece":           extraer_valor(texto, "etapa unece"),
        "textura":               extraer_valor(texto, "textura"),
        "manchas_oscuras":       extraer_valor(texto, "manchas oscuras"),
        "golpes_deformaciones":  extraer_valor(texto, "golpes"),
        "signos_enfermedad":     extraer_valor(texto, "signos de enfermedad"),
        "apto_nacional":         extraer_valor(texto, "exportaci.*nacional"),
        "apto_internacional":    extraer_valor(texto, "exportaci.*internacional"),
        "diagnostico":           extraer_valor(texto, "diagn"),
        "respuesta_completa":    texto
    }

def obtener_imagenes():
    """Obtiene lista de todas las imágenes del dataset."""
    imagenes = []
    for ext in EXTENSIONES:
        imagenes.extend(Path(DATASET_DIR).rglob(f"*{ext}"))
        imagenes.extend(Path(DATASET_DIR).rglob(f"*{ext.upper()}"))
    return sorted(set(imagenes))

def cargar_progreso():
    """Carga el CSV existente para retomar donde se dejó."""
    if os.path.exists(CSV_SALIDA):
        df = pd.read_csv(CSV_SALIDA)
        procesados = set(df["archivo"].tolist())
        return df, procesados
    return pd.DataFrame(columns=COLUMNAS), set()

def main():
    print("""
╔══════════════════════════════════════════╗
║   BananaCheck AI — Análisis en Lote     ║
╚══════════════════════════════════════════╝
""")

    # Verificar conexión con Ollama
    try:
        res = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        modelos = [m["name"] for m in res.json().get("models", [])]
        if not any(MODELO in m for m in modelos):
            print(f"❌ Modelo '{MODELO}' no encontrado. Instálalo con: ollama pull {MODELO}")
            return
        print(f"✅ Ollama conectado — modelo: {MODELO}")
    except Exception as e:
        print(f"❌ No se pudo conectar a Ollama: {e}")
        print("   Ejecuta: OLLAMA_ORIGINS=* ollama serve")
        return

    # Obtener imágenes
    imagenes = obtener_imagenes()
    total = len(imagenes)
    print(f"📁 Imágenes encontradas: {total}")
    print(f"💾 CSV de salida: {CSV_SALIDA}\n")

    if total == 0:
        print("❌ No se encontraron imágenes en:", DATASET_DIR)
        return

    # Cargar progreso anterior
    df, procesados = cargar_progreso()
    pendientes = [img for img in imagenes if os.path.basename(img) not in procesados]
    print(f"✅ Ya procesadas: {len(procesados)}")
    print(f"⏳ Pendientes:    {len(pendientes)}\n")

    if not pendientes:
        print("🎉 ¡Todas las imágenes ya fueron procesadas!")
        print(f"   CSV listo en: {CSV_SALIDA}")
        return

    # Procesar en lote
    errores = 0
    for i, ruta in enumerate(pendientes, 1):
        nombre = os.path.basename(ruta)
        print(f"[{i}/{len(pendientes)}] Analizando: {nombre} ...", end=" ", flush=True)

        try:
            respuesta = analizar_imagen(ruta)
            datos     = parsear_respuesta(respuesta, ruta)
            df = pd.concat([df, pd.DataFrame([datos])], ignore_index=True)

            # Guardar cada 10 imágenes
            if i % 10 == 0:
                df.to_csv(CSV_SALIDA, index=False)
                print(f"\n💾 Guardado automático — {i} procesadas\n", end="")

            print(f"✅ UNECE:{datos['etapa_unece']} | Nacional:{datos['apto_nacional']} | Internacional:{datos['apto_internacional']}")

        except Exception as e:
            errores += 1
            print(f"❌ Error: {e}")
            datos = {col: "" for col in COLUMNAS}
            datos["archivo"]    = nombre
            datos["fecha_hora"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            datos["diagnostico"] = f"ERROR: {str(e)[:100]}"
            df = pd.concat([df, pd.DataFrame([datos])], ignore_index=True)

    # Guardar CSV final
    df.to_csv(CSV_SALIDA, index=False)

    print(f"""
╔══════════════════════════════════════════╗
║   ✅ Análisis completado                 ║
╠══════════════════════════════════════════╣
║  Total procesadas: {total:<22}║
║  Errores:          {errores:<22}║
║  CSV guardado en:                        ║
║  dataset_analizado.csv                   ║
╚══════════════════════════════════════════╝
""")

if __name__ == "__main__":
    main()
