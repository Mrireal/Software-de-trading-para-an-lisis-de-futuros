# Software de Trading en Futuros

## Descripción

Este software es una herramienta completa para el análisis y trading de futuros y criptomonedas. Proporciona análisis técnico avanzado, recomendaciones de compra/venta personalizadas y una interfaz gráfica intuitiva para ayudar a los traders a tomar decisiones informadas.

## Características Principales

- **Obtención de datos en tiempo real**: Conexión con múltiples fuentes de datos financieros (Yahoo Finance, Binance)
- **Análisis técnico avanzado**: Más de 15 indicadores técnicos implementados
- **Sistema de recomendaciones inteligente**: Estrategias personalizadas según perfil de riesgo
- **Interfaz gráfica interactiva**: Visualización de datos y señales de trading
- **Backtesting de estrategias**: Evaluación del rendimiento histórico de diferentes estrategias
- **Personalización**: Adaptable a diferentes activos y perfiles de riesgo

## Requisitos del Sistema

- Python 3.8 o superior
- Dependencias: pandas, numpy, matplotlib, streamlit, plotly, requests

## Instalación

1. Clone el repositorio:
```
git clone https://github.com/usuario/trading_futures.git
cd trading_futures
```

2. Instale las dependencias:
```
pip install -r requirements.txt
```

## Uso

### Iniciar la aplicación

```
python app.py
```

Esto iniciará la interfaz web en http://localhost:8501

### Estructura de Archivos

- `app.py`: Script principal para iniciar la aplicación
- `data_module.py`: Módulo para obtención de datos financieros
- `analysis_module.py`: Módulo de análisis técnico
- `recommendation_module.py`: Sistema de recomendaciones
- `ui_module.py`: Interfaz de usuario con Streamlit
- `data_utils.py`: Utilidades para procesamiento de datos
- `test_software.py`: Script para pruebas exhaustivas

## Guía de Uso

### 1. Selección de Activo

En la barra lateral, seleccione el activo que desea analizar. Puede elegir entre criptomonedas (BTC, ETH, etc.) o acciones (AAPL, MSFT, etc.).

### 2. Configuración de Parámetros

- **Intervalo de tiempo**: Seleccione el intervalo para los datos (1m, 5m, 15m, 1h, 1d, etc.)
- **Período de análisis**: Elija el rango de fechas para el análisis
- **Perfil de riesgo**: Seleccione entre conservador, moderado o agresivo

### 3. Análisis y Recomendaciones

La aplicación mostrará:
- Gráfico de precios con indicadores técnicos
- Señales de compra/venta identificadas
- Recomendación actual con nivel de confianza
- Niveles sugeridos de entrada, stop loss y take profit

### 4. Backtesting

Utilice la pestaña de backtesting para evaluar el rendimiento histórico de diferentes estrategias con el activo seleccionado.

## Personalización

### Modificar Estrategias

Las estrategias de trading se pueden personalizar editando el archivo `strategies.json`. Cada estrategia define:
- Indicadores utilizados y sus parámetros
- Pesos para cada indicador
- Filtros para las señales
- Configuración de stop loss y take profit

### Añadir Nuevos Indicadores

Para añadir nuevos indicadores técnicos, modifique el archivo `analysis_module.py` implementando el cálculo del indicador y actualice la interfaz en `ui_module.py` para visualizarlo.

## Limitaciones

- El software proporciona recomendaciones basadas en análisis técnico, pero no garantiza resultados
- El trading de futuros y criptomonedas implica riesgos significativos
- Las APIs gratuitas pueden tener limitaciones en cuanto a frecuencia de actualización y disponibilidad

## Soporte

Para reportar problemas o solicitar nuevas características, por favor abra un issue en el repositorio de GitHub.

## Licencia

Este software se distribuye bajo la licencia MIT. Consulte el archivo LICENSE para más detalles.
