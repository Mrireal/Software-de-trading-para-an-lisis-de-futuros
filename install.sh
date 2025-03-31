#!/bin/bash

# Script de instalación para el Software de Trading en Futuros
# Este script instala todas las dependencias necesarias y configura el entorno

echo "=== Instalando Software de Trading en Futuros ==="
echo "Verificando requisitos previos..."

# Verificar Python
if command -v python3 &>/dev/null; then
    python_version=$(python3 --version | cut -d " " -f 2)
    echo "✅ Python $python_version encontrado"
else
    echo "❌ Python 3 no encontrado. Por favor, instale Python 3.8 o superior."
    exit 1
fi

# Crear entorno virtual
echo "Creando entorno virtual..."
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
echo "Instalando dependencias..."
pip install --upgrade pip
pip install pandas numpy matplotlib streamlit plotly requests

# Verificar instalación
echo "Verificando instalación..."
python3 -c "import pandas, numpy, matplotlib, streamlit, plotly, requests; print('✅ Todas las dependencias instaladas correctamente')"

# Crear directorios necesarios
echo "Configurando directorios..."
mkdir -p .cache
mkdir -p test_results
mkdir -p images

echo "=== Instalación completada ==="
echo "Para iniciar el software, ejecute:"
echo "source venv/bin/activate"
echo "python app.py"
echo ""
echo "Acceda a la interfaz web en: http://localhost:8501"
