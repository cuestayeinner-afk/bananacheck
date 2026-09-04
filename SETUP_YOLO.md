# BananaCheck con YOLOv8

## Instalación rápida

### 1. Dependencias básicas (existentes)
```bash
pip install pandas
```

### 2. Dependencias YOLOv8 (opcional pero recomendado)
```bash
pip install -r requirements_yolo.txt
```

Esto instala:
- **ultralytics**: YOLOv8 framework
- **opencv-python**: Procesamiento de imágenes
- **torch & torchvision**: Backend de ML

### 3. Arrancar servidor
```bash
python3 bananacheck_server.py
```

Abre en navegador:
```
http://127.0.0.1:9090/index.html
```

---

## Nuevas capacidades

### Detección YOLOv8
El servidor ahora tiene un endpoint para detección con YOLOv8:

```bash
curl -X POST \
  -F "imagen=@foto_banano.jpg" \
  http://127.0.0.1:9090/yolo/detectar
```

**Respuesta:**
```json
{
  "success": true,
  "banano_detectado": true,
  "confianza": 0.87,
  "detecciones": 1,
  "diagnostico": "Banano detectado ✓"
}
```

### Integración en HTML/JS
El `index.html` ahora puede usar YOLOv8 como validación adicional:
- Mantiene análisis con Ollama + heurísticas
- Agrega validación rápida con YOLOv8
- Rechaza imágenes que no sean bananos

---

## Archivos nuevos

- **yolo_inference.py**: Script de inferencia YOLOv8
- **requirements_yolo.txt**: Dependencias
- **bananacheck_server.py**: Servidor actualizado con endpoint `/yolo/detectar`

---

## Para usar modelo entrenado personalizado

Coloca tu modelo en:
```
~/Descargas/bananacheck/modelo_banano.pt
```

El sistema lo usará automáticamente. Si no existe, usa el modelo base `yolov8n-seg.pt`.

---

## Performance

- **YOLOv8n**: ~5-10ms por imagen (CPU)
- **Moondream (Ollama)**: ~500-2000ms por imagen (CPU)
- **Heurísticas**: ~1-2ms por imagen

**Flujo recomendado:**
1. YOLOv8 para validación rápida (¿es banano?)
2. Ollama para análisis detallado (calidad, defectos)
3. Heurísticas como fallback si Ollama falla
