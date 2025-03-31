import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
import json
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("web_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("web_app")

# Importar nuestros módulos
try:
    from data_module import DataManager
    from analysis_module import TechnicalAnalysis
    from recommendation_module import TradingRecommendationEngine
    import data_utils as utils
    from config import CONFIG, DEFAULT_SYMBOLS, TIME_INTERVALS, TIME_PERIODS, TECHNICAL_INDICATORS
except ImportError as e:
    logger.error(f"Error al importar módulos: {str(e)}")
    st.error(f"Error al importar módulos: {str(e)}")

# Configuración de la página
st.set_page_config(
    page_title="Trading Futures - Análisis y Recomendaciones",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #616161;
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
        font-weight: bold;
    }
    .stApp {
        background-color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar estado de sesión
if 'data_manager' not in st.session_state:
    try:
        st.session_state.data_manager = DataManager()
    except Exception as e:
        logger.error(f"Error al inicializar DataManager: {str(e)}")
        st.error(f"Error al inicializar DataManager: {str(e)}")

if 'analyzer' not in st.session_state:
    try:
        st.session_state.analyzer = TechnicalAnalysis()
    except Exception as e:
        logger.error(f"Error al inicializar TechnicalAnalysis: {str(e)}")
        st.error(f"Error al inicializar TechnicalAnalysis: {str(e)}")

if 'recommendation_engine' not in st.session_state:
    try:
        st.session_state.recommendation_engine = TradingRecommendationEngine()
    except Exception as e:
        logger.error(f"Error al inicializar TradingRecommendationEngine: {str(e)}")
        st.error(f"Error al inicializar TradingRecommendationEngine: {str(e)}")

if 'current_data' not in st.session_state:
    st.session_state.current_data = None

if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

# Funciones auxiliares
def load_data(symbol, interval, period):
    """Carga datos para el símbolo e intervalo especificados"""
    try:
        # Calcular fecha de inicio según el período
        end_date = datetime.now()
        
        if period == "1d":
            start_date = end_date - timedelta(days=1)
        elif period == "5d":
            start_date = end_date - timedelta(days=5)
        elif period == "1mo":
            start_date = end_date - timedelta(days=30)
        elif period == "3mo":
            start_date = end_date - timedelta(days=90)
        elif period == "6mo":
            start_date = end_date - timedelta(days=180)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "2y":
            start_date = end_date - timedelta(days=730)
        elif period == "5y":
            start_date = end_date - timedelta(days=1825)
        else:  # max
            start_date = end_date - timedelta(days=3650)  # 10 años
        
        start_date_str = start_date.strftime('%Y-%m-%d')
        
        # Obtener datos
        df = st.session_state.data_manager.get_historical_data(
            symbol=symbol,
            interval=interval,
            start_time=start_date_str
        )
        
        if df is None or df.empty:
            st.error(f"No se pudieron obtener datos para {symbol}")
            return None
        
        # Limpiar datos
        df = utils.clean_dataframe(df)
        
        # Aplicar análisis técnico
        df = st.session_state.analyzer.add_all_indicators(df)
        
        # Generar señales
        df = st.session_state.analyzer.generate_signals(df)
        
        # Actualizar estado de sesión
        st.session_state.current_data = df
        st.session_state.last_update = datetime.now()
        
        return df
    
    except Exception as e:
        logger.error(f"Error al cargar datos: {str(e)}")
        st.error(f"Error al cargar datos: {str(e)}")
        return None

def create_candlestick_chart(df, indicators=None):
    """Crea un gráfico de velas con indicadores seleccionados"""
    if df is None or df.empty:
        return None
    
    # Crear figura
    fig = go.Figure()
    
    # Añadir gráfico de velas
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Precio",
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350'
    ))
    
    # Añadir indicadores seleccionados
    if indicators:
        if 'sma' in indicators:
            for period in [20, 50, 200]:
                col = f'sma_{period}'
                if col in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df[col],
                        mode='lines',
                        name=f'SMA {period}',
                        line=dict(width=1.5)
                    ))
        
        if 'ema' in indicators:
            for period in [12, 26, 50]:
                col = f'ema_{period}'
                if col in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df[col],
                        mode='lines',
                        name=f'EMA {period}',
                        line=dict(width=1.5, dash='dash')
                    ))
        
        if 'bollinger' in indicators:
            if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['bb_upper'],
                    mode='lines',
                    name='Bollinger Superior',
                    line=dict(width=1, color='rgba(173, 216, 230, 0.7)')
                ))
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['bb_middle'],
                    mode='lines',
                    name='Bollinger Media',
                    line=dict(width=1, color='rgba(173, 216, 230, 1)')
                ))
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['bb_lower'],
                    mode='lines',
                    name='Bollinger Inferior',
                    line=dict(width=1, color='rgba(173, 216, 230, 0.7)')
                ))
                
                # Rellenar área entre bandas
                fig.add_trace(go.Scatter(
                    x=df['timestamp'].tolist() + df['timestamp'].tolist()[::-1],
                    y=df['bb_upper'].tolist() + df['bb_lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(173, 216, 230, 0.2)',
                    line=dict(color='rgba(255, 255, 255, 0)'),
                    hoverinfo='skip',
                    showlegend=False
                ))
    
    # Añadir señales de compra/venta
    buy_signals = df[df['signal'] == 1]
    sell_signals = df[df['signal'] == -1]
    
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals['timestamp'],
            y=buy_signals['low'] * 0.99,  # Ligeramente por debajo del precio
            mode='markers',
            name='Señal de Compra',
            marker=dict(
                symbol='triangle-up',
                size=10,
                color='green',
                line=dict(width=1, color='darkgreen')
            )
        ))
    
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals['timestamp'],
            y=sell_signals['high'] * 1.01,  # Ligeramente por encima del precio
            mode='markers',
            name='Señal de Venta',
            marker=dict(
                symbol='triangle-down',
                size=10,
                color='red',
                line=dict(width=1, color='darkred')
            )
        ))
    
    # Configurar diseño
    fig.update_layout(
        title=f'Análisis Técnico',
        xaxis_title='Fecha',
        yaxis_title='Precio',
        height=600,
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Configurar ejes
    fig.update_xaxes(
        rangeslider_visible=False,
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # Ocultar fines de semana
        ]
    )
    
    return fig

def create_indicator_charts(df):
    """Crea gráficos para indicadores técnicos"""
    if df is None or df.empty:
        return None
    
    charts = []
    
    # RSI
    if 'rsi' in df.columns:
        fig_rsi = go.Figure()
        
        fig_rsi.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['rsi'],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=1.5)
        ))
        
        # Añadir líneas de referencia
        fig_rsi.add_shape(
            type="line",
            x0=df['timestamp'].iloc[0],
            y0=70,
            x1=df['timestamp'].iloc[-1],
            y1=70,
            line=dict(color="red", width=1, dash="dash"),
        )
        
        fig_rsi.add_shape(
            type="line",
            x0=df['timestamp'].iloc[0],
            y0=30,
            x1=df['timestamp'].iloc[-1],
            y1=30,
            line=dict(color="green", width=1, dash="dash"),
        )
        
        fig_rsi.update_layout(
            title='RSI (Índice de Fuerza Relativa)',
            xaxis_title='Fecha',
            yaxis_title='RSI',
            height=250,
            template='plotly_white',
            yaxis=dict(range=[0, 100])
        )
        
        charts.append(fig_rsi)
    
    # MACD
    if all(col in df.columns for col in ['macd', 'macd_signal', 'macd_histogram']):
        fig_macd = go.Figure()
        
        fig_macd.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['macd'],
            mode='lines',
            name='MACD',
            line=dict(color='blue', width=1.5)
        ))
        
        fig_macd.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['macd_signal'],
            mode='lines',
            name='Señal',
            line=dict(color='red', width=1.5)
        ))
        
        # Histograma
        colors = ['green' if val >= 0 else 'red' for val in df['macd_histogram']]
        
        fig_macd.add_trace(go.Bar(
            x=df['timestamp'],
            y=df['macd_histogram'],
            name='Histograma',
            marker_color=colors
        ))
        
        fig_macd.update_layout(
            title='MACD (Convergencia/Divergencia de Medias Móviles)',
            xaxis_title='Fecha',
            yaxis_title='Valor',
            height=250,
            template='plotly_white'
        )
        
        charts.append(fig_macd)
    
    # Volumen
    if 'volume' in df.columns:
        fig_volume = go.Figure()
        
        # Determinar colores basados en cambio de precio
        colors = ['green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red' 
                 for i in range(len(df))]
        
        fig_volume.add_trace(go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volumen',
            marker_color=colors
        ))
        
        # Añadir media móvil de volumen si está disponible
        if 'volume_sma' in df.columns:
            fig_volume.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['volume_sma'],
                mode='lines',
                name='Media Volumen',
                line=dict(color='blue', width=1.5)
            ))
        
        fig_volume.update_layout(
            title='Volumen',
            xaxis_title='Fecha',
            yaxis_title='Volumen',
            height=200,
            template='plotly_white'
        )
        
        charts.append(fig_volume)
    
    return charts

def format_recommendation(recommendation):
    """Formatea la recomendación para mostrarla en la interfaz"""
    if recommendation is None:
        return None
    
    rec_type = recommendation["recommendation"]
    confidence = recommendation["confidence"]
    entry_price = recommendation["entry_price"]
    stop_loss = recommendation["stop_loss"]
    take_profit = recommendation["take_profit"]
    reason = recommendation["reason"]
    
    # Determinar clase CSS según tipo de recomendación
    if rec_type == "COMPRAR":
        rec_class = "buy-signal"
    elif rec_type == "VENDER":
        rec_class = "sell-signal"
    else:
        rec_class = "neutral-signal"
    
    # Formatear HTML
    html = f"""
    <div class="info-box">
        <h3>Recomendación: <span class="{rec_class}">{rec_type}</span></h3>
        <p><strong>Confianza:</strong> {confidence}%</p>
        <p><strong>Precio de entrada:</strong> ${entry_price:.2f if entry_price else 'N/A'}</p>
    """
    
    if rec_type != "NEUTRAL" and stop_loss and take_profit:
        html += f"""
        <p><strong>Stop Loss:</strong> ${stop_loss:.2f}</p>
        <p><strong>Take Profit:</strong> ${take_profit:.2f}</p>
        """
    
    html += f"""
        <p><strong>Razón:</strong> {reason}</p>
    </div>
    """
    
    return html

def display_metrics(df, symbol):
    """Muestra métricas clave del activo"""
    if df is None or df.empty:
        return
    
    # Obtener datos recientes
    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) > 1 else latest
    
    # Calcular métricas
    current_price = latest['close']
    price_change = current_price - previous['close']
    price_change_pct = (price_change / previous['close']) * 100
    
    # Calcular rango de precios
    price_high = df['high'].max()
    price_low = df['low'].min()
    
    # Calcular volumen promedio
    avg_volume = df['volume'].mean()
    
    # Crear columnas para métricas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsaf
(Content truncated due to size limit. Use line ranges to read in chunks)