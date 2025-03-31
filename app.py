"""
Script para iniciar la interfaz de usuario del Software de Trading en Futuros.
Este script configura el entorno y lanza la aplicación Streamlit.
"""

import os
import sys
import subprocess
import logging
from datetime import datetime

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("app_launcher")

def check_dependencies():
    """Verifica que todas las dependencias estén instaladas"""
    required_packages = [
        "streamlit",
        "pandas",
        "numpy",
        "matplotlib",
        "plotly",
        "requests"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Faltan las siguientes dependencias: {', '.join(missing_packages)}")
        logger.info("Instalando dependencias faltantes...")
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
            logger.info("Dependencias instaladas correctamente")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error al instalar dependencias: {str(e)}")
            return False
    
    return True

def create_example_image():
    """Crea una imagen de ejemplo para la interfaz si no existe"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Verificar si el directorio de imágenes existe
    if not os.path.exists("images"):
        os.makedirs("images")
    
    # Ruta de la imagen de ejemplo
    image_path = "images/example_chart.png"
    
    # Crear imagen solo si no existe
    if not os.path.exists(image_path):
        # Crear datos de ejemplo
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.normal(0, 0.1, 100)
        
        # Crear gráfico
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label='Precio')
        plt.title('Ejemplo de Análisis Técnico')
        plt.xlabel('Tiempo')
        plt.ylabel('Precio')
        plt.grid(True)
        plt.legend()
        
        # Guardar imagen
        plt.savefig(image_path)
        plt.close()
        
        logger.info(f"Imagen de ejemplo creada en {image_path}")
    
    return image_path

def launch_app():
    """Inicia la aplicación Streamlit"""
    try:
        logger.info("Iniciando aplicación Streamlit...")
        
        # Comando para iniciar Streamlit
        cmd = [
            "streamlit", "run", "ui_module.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ]
        
        # Ejecutar comando
        process = subprocess.Popen(cmd)
        
        logger.info("Aplicación Streamlit iniciada correctamente")
        logger.info("Accede a la aplicación en http://localhost:8501")
        
        return process
        
    except Exception as e:
        logger.error(f"Error al iniciar la aplicación: {str(e)}")
        return None

def main():
    """Función principal"""
    logger.info("Iniciando Software de Trading en Futuros")
    
    # Verificar dependencias
    if not check_dependencies():
        logger.error("No se pudieron instalar todas las dependencias. Abortando.")
        return
    
    # Crear imagen de ejemplo
    create_example_image()
    
    # Iniciar aplicación
    app_process = launch_app()
    
    if app_process:
        try:
            # Mantener la aplicación en ejecución
            app_process.wait()
        except KeyboardInterrupt:
            logger.info("Aplicación detenida por el usuario")
            app_process.terminate()
    
    logger.info("Software de Trading en Futuros finalizado")

if __name__ == "__main__":
    main()
