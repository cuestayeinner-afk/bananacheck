#!/bin/bash
# Setup automático para BananaCheck con YOLOv8

set -e

echo "╔════════════════════════════════════════╗"
echo "║  BananaCheck AI — Setup YOLOv8         ║"
echo "╚════════════════════════════════════════╝"

# Detectar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 no instalado"
    exit 1
fi

PYTHON=$(which python3)
echo "✅ Python: $PYTHON"

# Crear venv si no existe
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual..."
    python3 -m venv venv
fi

# Activar venv
echo "🔧 Activando entorno virtual..."
source venv/bin/activate

# Actualizar pip
echo "📥 Actualizando pip..."
pip install --upgrade pip --quiet

# Instalar requisitos básicos
echo "📥 Instalando requisitos básicos..."
pip install pandas --quiet

# Preguntar si instalar YOLOv8
echo ""
echo "⚠️  YOLOv8 requiere descargas grandes (PyTorch, modelos, etc.)"
read -p "¿Instalar YOLOv8? (s/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Ss]$ ]]; then
    echo "📥 Instalando YOLOv8 y dependencias..."
    echo "   (Esto puede tomar 5-10 minutos...)"
    pip install -r requirements_yolo.txt --quiet
    echo "✅ YOLOv8 instalado"
else
    echo "⏭️  YOLOv8 omitido — server funcionará sin detección YOLOv8"
fi

echo ""
echo "╔════════════════════════════════════════╗"
echo "║  ✅ Setup completado                   ║"
echo "╚════════════════════════════════════════╝"
echo ""
echo "📌 Para arrancar:"
echo "   source venv/bin/activate"
echo "   python3 bananacheck_server.py"
echo ""
echo "🌐 Luego abre:"
echo "   http://127.0.0.1:9090/index.html"
echo ""
