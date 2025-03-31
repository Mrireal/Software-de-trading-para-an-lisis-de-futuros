# Trading Futures Web App

Este repositorio contiene una aplicación web para análisis y recomendaciones de trading en futuros y criptomonedas.

## Descripción

La aplicación proporciona herramientas para analizar activos financieros y recibir recomendaciones de trading basadas en análisis técnico. Incluye gráficos interactivos, indicadores técnicos, señales de trading y backtesting de estrategias.

## Características

- **Análisis Técnico**: Visualización de gráficos de precios con múltiples indicadores técnicos
- **Señales de Trading**: Identificación de oportunidades de compra y venta
- **Recomendaciones**: Sugerencias personalizadas según perfil de riesgo
- **Backtesting**: Evaluación del rendimiento histórico de diferentes estrategias

## Requisitos

- Python 3.8 o superior
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Plotly
- Requests

## Instalación Local

1. Clonar el repositorio:
```
git clone https://github.com/yourusername/trading-futures-app.git
cd trading-futures-app
```

2. Instalar dependencias:
```
pip install -r requirements.txt
```

3. Ejecutar la aplicación:
```
streamlit run streamlit_app.py
```

## Despliegue en Streamlit Cloud

Esta aplicación está configurada para ser desplegada en Streamlit Cloud. Para desplegarla:

1. Crear una cuenta en [Streamlit Cloud](https://streamlit.io/cloud)
2. Conectar tu repositorio de GitHub
3. Seleccionar el archivo `streamlit_app.py` como punto de entrada
4. Configurar los requisitos según el archivo `requirements.txt`

## Estructura del Proyecto

- `streamlit_app.py`: Aplicación principal de Streamlit
- `streamlit_config.yaml`: Configuración para despliegue en Streamlit Cloud
- `requirements.txt`: Dependencias del proyecto

## Uso

1. Seleccionar la categoría de activo (Criptomonedas, Acciones, Futuros)
2. Elegir un símbolo específico
3. Configurar el intervalo de tiempo y período
4. Seleccionar el perfil de riesgo
5. Activar los indicadores técnicos deseados
6. Hacer clic en "Cargar Datos" para visualizar el análisis

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para más detalles.

## Contacto

Para preguntas o sugerencias, por favor abrir un issue en este repositorio.
