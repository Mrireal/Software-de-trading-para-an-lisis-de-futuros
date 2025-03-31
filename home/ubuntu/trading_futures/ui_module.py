"""
Módulo de interfaz de usuario para el Software de Trading en Futuros

Este módulo implementa una interfaz web interactiva utilizando Streamlit
para visualizar datos financieros, análisis técnico y recomendaciones de trading.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import logging

# Importar nuestros módulos
from data_module import DataManager
from analysis_module import TechnicalAnalysis
import data_utils as utils

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_ui.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ui_module")

# Configuración de la página
def setup_page():
    """Configura la página de Streamlit"""
    st.set_page_config(
        page_title="Software de Trading en Futuros",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Título principal
    st.title("📊 Software de Trading en Futuros")
    st.markdown("*Análisis técnico y recomendaciones para trading en futuros y criptomonedas*")
    
    # Estilo personalizado
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .buy-signal {
        color: #4CAF50;
        font-weight: bold;
    }
    .sell-signal {
        color: #F44336;
        font-weight: bold;
    }
    .neutral-signal {
        color: #9E9E9E;
    }
    </style>
    """, unsafe_allow_html=True)

def sidebar_filters():
    """Crea los filtros de la barra lateral"""
    st.sidebar.header("Configuración")
    
    # Selección de proveedor de datos
    data_provider = st.sidebar.selectbox(
        "Proveedor de datos",
        options=["Yahoo Finance", "Binance"],
        index=0
    )
    
    # Mapeo de nombres amigables a identificadores internos
    provider_map = {
        "Yahoo Finance": "yahoo",
        "Binance": "binance"
    }
    
    # Obtener lista de símbolos según el proveedor
    data_manager = DataManager(default_provider=provider_map[data_provider])
    
    # Tipo de activo
    asset_type = st.sidebar.selectbox(
        "Tipo de activo",
        options=["Criptomonedas", "Acciones", "Futuros"],
        index=0
    )
    
    # Mapeo de tipos de activos
    asset_type_map = {
        "Criptomonedas": "crypto",
        "Acciones": "stock",
        "Futuros": "futures"
    }
    
    # Obtener símbolos disponibles
    symbols = data_manager.get_symbols_list(asset_type=asset_type_map[asset_type])
    
    # Si no hay símbolos, usar una lista predefinida
    if not symbols:
        if asset_type == "Criptomonedas":
            symbols = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"]
        elif asset_type == "Acciones":
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        else:  # Futuros
            symbols = ["ES=F", "NQ=F", "YM=F", "GC=F", "CL=F"]
    
    # Permitir entrada manual o selección de lista
    symbol_input_method = st.sidebar.radio(
        "Método de selección de símbolo",
        options=["Seleccionar de lista", "Ingresar manualmente"],
        index=0
    )
    
    if symbol_input_method == "Seleccionar de lista":
        symbol = st.sidebar.selectbox("Símbolo", options=symbols)
    else:
        symbol = st.sidebar.text_input("Símbolo", value=symbols[0] if symbols else "BTC-USD")
    
    # Intervalo de tiempo
    interval = st.sidebar.selectbox(
        "Intervalo",
        options=["1 minuto", "5 minutos", "15 minutos", "30 minutos", "1 hora", "4 horas", "1 día", "1 semana", "1 mes"],
        index=6  # Por defecto: 1 día
    )
    
    # Mapeo de intervalos
    interval_map = {
        "1 minuto": "1m",
        "5 minutos": "5m",
        "15 minutos": "15m",
        "30 minutos": "30m",
        "1 hora": "1h",
        "4 horas": "4h",
        "1 día": "1d",
        "1 semana": "1wk",
        "1 mes": "1mo"
    }
    
    # Período de tiempo
    period = st.sidebar.selectbox(
        "Período",
        options=["1 semana", "1 mes", "3 meses", "6 meses", "1 año", "2 años", "5 años", "Máximo"],
        index=2  # Por defecto: 3 meses
    )
    
    # Calcular fecha de inicio según el período
    start_date = None
    if period == "1 semana":
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    elif period == "1 mes":
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    elif period == "3 meses":
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    elif period == "6 meses":
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    elif period == "1 año":
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    elif period == "2 años":
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    elif period == "5 años":
        start_date = (datetime.now() - timedelta(days=1825)).strftime('%Y-%m-%d')
    else:  # Máximo
        start_date = "2010-01-01"
    
    # Indicadores técnicos a mostrar
    st.sidebar.header("Indicadores Técnicos")
    
    show_sma = st.sidebar.checkbox("Medias Móviles Simples (SMA)", value=True)
    show_ema = st.sidebar.checkbox("Medias Móviles Exponenciales (EMA)", value=False)
    show_bollinger = st.sidebar.checkbox("Bandas de Bollinger", value=True)
    show_macd = st.sidebar.checkbox("MACD", value=True)
    show_rsi = st.sidebar.checkbox("RSI", value=True)
    show_stochastic = st.sidebar.checkbox("Oscilador Estocástico", value=False)
    show_atr = st.sidebar.checkbox("ATR", value=False)
    show_adx = st.sidebar.checkbox("ADX", value=False)
    show_ichimoku = st.sidebar.checkbox("Ichimoku Cloud", value=False)
    show_volume = st.sidebar.checkbox("Indicadores de Volumen", value=True)
    
    # Parámetros de indicadores
    st.sidebar.header("Parámetros de Indicadores")
    
    # Parámetros de SMA
    sma_periods = []
    if show_sma:
        sma_col1, sma_col2 = st.sidebar.columns(2)
        with sma_col1:
            sma1 = st.number_input("SMA 1", min_value=1, max_value=200, value=20)
            sma_periods.append(sma1)
        with sma_col2:
            sma2 = st.number_input("SMA 2", min_value=1, max_value=200, value=50)
            sma_periods.append(sma2)
    
    # Parámetros de EMA
    ema_periods = []
    if show_ema:
        ema_col1, ema_col2 = st.sidebar.columns(2)
        with ema_col1:
            ema1 = st.number_input("EMA 1", min_value=1, max_value=200, value=12)
            ema_periods.append(ema1)
        with ema_col2:
            ema2 = st.number_input("EMA 2", min_value=1, max_value=200, value=26)
            ema_periods.append(ema2)
    
    # Parámetros de Bollinger Bands
    bb_period = 20
    bb_std_dev = 2.0
    if show_bollinger:
        bb_col1, bb_col2 = st.sidebar.columns(2)
        with bb_col1:
            bb_period = st.number_input("Período BB", min_value=1, max_value=100, value=20)
        with bb_col2:
            bb_std_dev = st.number_input("Desviaciones Estándar", min_value=0.1, max_value=5.0, value=2.0, step=0.1)
    
    # Parámetros de RSI
    rsi_period = 14
    if show_rsi:
        rsi_period = st.sidebar.number_input("Período RSI", min_value=1, max_value=100, value=14)
    
    # Botón para generar análisis
    st.sidebar.header("Acciones")
    generate_button = st.sidebar.button("Generar Análisis", type="primary")
    
    # Devolver todos los parámetros
    return {
        "provider": provider_map[data_provider],
        "symbol": symbol,
        "interval": interval_map[interval],
        "start_date": start_date,
        "indicators": {
            "sma": {"show": show_sma, "periods": sma_periods},
            "ema": {"show": show_ema, "periods": ema_periods},
            "bollinger": {"show": show_bollinger, "period": bb_period, "std_dev": bb_std_dev},
            "macd": {"show": show_macd},
            "rsi": {"show": show_rsi, "period": rsi_period},
            "stochastic": {"show": show_stochastic},
            "atr": {"show": show_atr},
            "adx": {"show": show_adx},
            "ichimoku": {"show": show_ichimoku},
            "volume": {"show": show_volume}
        },
        "generate": generate_button
    }

def load_data(params):
    """
    Carga datos financieros según los parámetros especificados
    
    Args:
        params (dict): Parámetros de configuración
        
    Returns:
        pandas.DataFrame: DataFrame con datos financieros
    """
    # Mostrar mensaje de carga
    with st.spinner(f"Cargando datos para {params['symbol']}..."):
        try:
            # Crear gestor de datos
            data_manager = DataManager(default_provider=params['provider'])
            
            # Obtener datos históricos
            df = data_manager.get_historical_data(
                symbol=params['symbol'],
                interval=params['interval'],
                start_time=params['start_date']
            )
            
            # Verificar si se obtuvieron datos
            if df is None or df.empty:
                st.error(f"No se pudieron obtener datos para {params['symbol']}. Por favor, verifica el símbolo e intenta nuevamente.")
                return None
            
            # Limpiar datos
            df = utils.clean_dataframe(df)
            
            # Obtener precio actual
            current_price = data_manager.get_current_price(params['symbol'])
            
            # Mostrar mensaje de éxito
            st.success(f"Datos cargados correctamente: {len(df)} registros")
            
            return df, current_price
            
        except Exception as e:
            st.error(f"Error al cargar datos: {str(e)}")
            logger.error(f"Error al cargar datos: {str(e)}")
            return None, None

def apply_technical_analysis(df, params):
    """
    Aplica análisis técnico al DataFrame según los parámetros especificados
    
    Args:
        df (pandas.DataFrame): DataFrame con datos financieros
        params (dict): Parámetros de configuración
        
    Returns:
        pandas.DataFrame: DataFrame con indicadores técnicos añadidos
    """
    # Mostrar mensaje de carga
    with st.spinner("Aplicando análisis técnico..."):
        try:
            # Crear analizador técnico
            analyzer = TechnicalAnalysis()
            
            # Aplicar indicadores según configuración
            if params['indicators']['sma']['show'] and params['indicators']['sma']['periods']:
                df = analyzer.add_moving_averages(df, windows=params['indicators']['sma']['periods'])
            
            if params['indicators']['ema']['show'] and params['indicators']['ema']['periods']:
                df = analyzer.add_exponential_moving_averages(df, windows=params['indicators']['ema']['periods'])
            
            if params['indicators']['bollinger']['show']:
                df = analyzer.add_bollinger_bands(
                    df, 
                    period=params['indicators']['bollinger']['period'],
                    std_dev=params['indicators']['bollinger']['std_dev']
                )
            
            if params['indicators']['macd']['show']:
                df = analyzer.add_macd(df)
            
            if params['indicators']['rsi']['show']:
                df = analyzer.add_rsi(df, period=params['indicators']['rsi']['period'])
            
            if params['indicators']['stochastic']['show']:
                df = analyzer.add_stochastic_oscillator(df)
            
            if params['indicators']['atr']['show']:
                df = analyzer.add_atr(df)
            
            if params['indicators']['adx']['show']:
                df = analyzer.add_adx(df)
            
            if params['indicators']['ichimoku']['show']:
                df = analyzer.add_ichimoku_cloud(df)
            
            if params['indicators']['volume']['show']:
                df = analyzer.add_volume_indicators(df)
            
            # Generar señales
            df = analyzer.generate_signals(df)
            
            return df
            
        except Exception as e:
            st.error(f"Error al aplicar análisis técnico: {str(e)}")
            logger.error(f"Error al aplicar análisis técnico: {str(e)}")
            return df

def display_price_info(df, current_price, symbol):
    """
    Muestra información básica sobre el precio
    
    Args:
        df (pandas.DataFrame): DataFrame con datos financieros
        current_price (float): Precio actual
        symbol (str): Símbolo del activo
    """
    # Crear columnas para mostrar información
    col1, col2, col3, col4 = st.columns(4)
    
    # Calcular métricas
    if df is not None and not df.empty:
        # Precio actual
        with col1:
            st.metric(
                label="Precio Actual",
                value=f"${current_price:.2f}" if current_price else "N/A",
                delta=f"{(current_price - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100:.2f}%" if current_price and len(df) > 1 else None
            )
        
        # Cambio en 24h
        with col2:
            if len(df) > 1:
                price_24h_ago = df[df['timestamp'] >= (datetime.now() - timedelta(days=1))].iloc[0]['close'] if not df[df['timestamp'] >= (datetime.now() - timedelta(days=1))].empty else df.iloc[0]['close']
                change_24h = (current_price - price_24h_ago) / price_24h_ago * 100 if current_price else 0
                st.metric(
                    label="Cambio 24h",
                    value=f"{change_24h:.2f}%",
                    delta=f"{change_24h:.2f}%",
                    delta_color="normal"
                )
            else:
                st.metric(label="Cambio 24h", value="N/A")
        
        # Rango de precio
        with col3:
            if len(df) > 0:
                price_low = df['low'].min()
                price_high = df['high'].max()
                st.metric(
                    label="Rango de Precio",
                    value=f"${price_low:.2f} - ${price_high:.2f}"
                )
            else:
                st.metric(label="Rango de Precio", value="N/A")
        
        # Volumen promedio
        with col4:
            if len(df) > 0 and 'volume' in df.columns:
                avg_volume = df['volume'].mean()
                st.metric(
                    label="Volumen Promedio",
                    value=f"{avg_volume:.2f}"
                )
            else:
                st.metric(label="Volumen Promedio", value="N/A")

def plot_candlestick_chart(df, symbol, indicators):
    """
    Genera un gráfico de velas con indicadores técnicos
    
    Args:
        df (pandas.DataFrame): DataFrame con datos financieros e indicadores
        symbol (str): Símbolo del activo
        indicators (dict): Configuración de indicadores a mostrar
    """
    if df is None or df.empty:
        st.warning("No hay datos suficientes para generar el gráfico")
        return
    
    # Crear figura con subplots
    fig = make_subplots(
        rows=3, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"Precio de {symbol}", "Volumen", "Indicadores")
    )
    
    # Añadir gráfico de velas
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Precio"
        ),
        row
(Content truncated due to size limit. Use line ranges to read in chunks)