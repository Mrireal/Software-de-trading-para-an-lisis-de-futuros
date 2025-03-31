#!/bin/bash

# Script para preparar y ejecutar la aplicación web de Trading Futures
# Este script configura el entorno y lanza la aplicación Streamlit para despliegue

echo "=== Preparando aplicación web de Trading Futures ==="

# Verificar Python
if command -v python3 &>/dev/null; then
    python_version=$(python3 --version | cut -d " " -f 2)
    echo "✅ Python $python_version encontrado"
else
    echo "❌ Python 3 no encontrado. Por favor, instale Python 3.8 o superior."
    exit 1
fi

# Instalar dependencias
echo "Instalando dependencias..."
pip install --upgrade pip
pip install pandas numpy matplotlib streamlit plotly requests

# Crear directorios necesarios
echo "Configurando directorios..."
mkdir -p .cache
mkdir -p test_results
mkdir -p images

# Configurar para despliegue
echo "Configurando para despliegue web..."

# Crear archivo de configuración para Streamlit
mkdir -p .streamlit
cat > .streamlit/config.toml << EOL
[server]
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
serverAddress = "0.0.0.0"
serverPort = 8501
gatherUsageStats = false

[theme]
primaryColor = "#1E88E5"
backgroundColor = "#FAFAFA"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#424242"
font = "sans serif"
EOL

echo "=== Configuración completada ==="
echo "Para iniciar la aplicación web, ejecute:"
echo "streamlit run web_app.py --server.port 8501 --server.address 0.0.0.0"
echo ""
echo "Para despliegue en producción, use:"
echo "nohup streamlit run web_app.py --server.port 8501 --server.address 0.0.0.0 &"
