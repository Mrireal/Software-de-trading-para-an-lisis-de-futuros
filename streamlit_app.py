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
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("web_app")

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

# Definir símbolos disponibles
DEFAULT_SYMBOLS = {
    "crypto": [
        {"symbol": "BTC-USD", "name": "Bitcoin"},
        {"symbol": "ETH-USD", "name": "Ethereum"},
        {"symbol": "SOL-USD", "name": "Solana"},
        {"symbol": "ADA-USD", "name": "Cardano"},
        {"symbol": "DOT-USD", "name": "Polkadot"},
        {"symbol": "DOGE-USD", "name": "Dogecoin"},
        {"symbol": "AVAX-USD", "name": "Avalanche"},
        {"symbol": "MATIC-USD", "name": "Polygon"},
        {"symbol": "LINK-USD", "name": "Chainlink"},
        {"symbol": "XRP-USD", "name": "Ripple"}
    ],
    "stocks": [
        {"symbol": "AAPL", "name": "Apple Inc."},
        {"symbol": "MSFT", "name": "Microsoft Corporation"},
        {"symbol": "GOOGL", "name": "Alphabet Inc."},
        {"symbol": "AMZN", "name": "Amazon.com Inc."},
        {"symbol": "META", "name": "Meta Platforms Inc."},
        {"symbol": "TSLA", "name": "Tesla Inc."},
        {"symbol": "NVDA", "name": "NVIDIA Corporation"},
        {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
        {"symbol": "V", "name": "Visa Inc."},
        {"symbol": "WMT", "name": "Walmart Inc."}
    ],
    "futures": [
        {"symbol": "ES=F", "name": "E-mini S&P 500"},
        {"symbol": "NQ=F", "name": "E-mini NASDAQ 100"},
        {"symbol": "YM=F", "name": "Mini Dow Jones"},
        {"symbol": "RTY=F", "name": "E-mini Russell 2000"},
        {"symbol": "GC=F", "name": "Gold"},
        {"symbol": "SI=F", "name": "Silver"},
        {"symbol": "CL=F", "name": "Crude Oil"},
        {"symbol": "NG=F", "name": "Natural Gas"},
        {"symbol": "ZC=F", "name": "Corn"},
        {"symbol": "ZW=F", "name": "Wheat"}
    ]
}

# Intervalos de tiempo disponibles
TIME_INTERVALS = [
    {"value": "1m", "name": "1 minuto"},
    {"value": "5m", "name": "5 minutos"},
    {"value": "15m", "name": "15 minutos"},
    {"value": "30m", "name": "30 minutos"},
    {"value": "1h", "name": "1 hora"},
    {"value": "4h", "name": "4 horas"},
    {"value": "1d", "name": "1 día"},
    {"value": "1wk", "name": "1 semana"},
    {"value": "1mo", "name": "1 mes"}
]

# Períodos de tiempo disponibles
TIME_PERIODS = [
    {"value": "1d", "name": "1 día"},
    {"value": "5d", "name": "5 días"},
    {"value": "1mo", "name": "1 mes"},
    {"value": "3mo", "name": "3 meses"},
    {"value": "6mo", "name": "6 meses"},
    {"value": "1y", "name": "1 año"},
    {"value": "2y", "name": "2 años"},
    {"value": "5y", "name": "5 años"},
    {"value": "max", "name": "Máximo"}
]

# Datos de ejemplo para demostración
@st.cache_data
def generate_sample_data(symbol, days=90):
    np.random.seed(42)  # Para reproducibilidad
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Generar fechas
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generar precios
    if symbol.startswith("BTC"):
        base_price = 50000
        volatility = 0.03
    elif symbol.startswith("ETH"):
        base_price = 3000
        volatility = 0.04
    elif symbol in ["AAPL", "MSFT", "GOOGL"]:
        base_price = 150
        volatility = 0.015
    else:
        base_price = 100
        volatility = 0.02
    
    # Generar movimiento de precios
    returns = np.random.normal(0.0005, volatility, size=len(date_range))
    price_factor = (1 + returns).cumprod()
    
    close_prices = base_price * price_factor
    
    # Generar OHLC
    daily_volatility = volatility / np.sqrt(252)
    high_prices = close_prices * (1 + np.random.uniform(0, daily_volatility * 2, size=len(date_range)))
    low_prices = close_prices * (1 - np.random.uniform(0, daily_volatility * 2, size=len(date_range)))
    open_prices = low_prices + np.random.uniform(0, 1, size=len(date_range)) * (high_prices - low_prices)
    
    # Generar volumen
    volume = np.random.lognormal(mean=np.log(1000000), sigma=1, size=len(date_range))
    
    # Crear DataFrame
    df = pd.DataFrame({
        'timestamp': date_range,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })
    
    # Añadir indicadores técnicos
    df = add_technical_indicators(df)
    
    return df

def add_technical_indicators(df):
    """Añade indicadores técnicos al DataFrame"""
    # SMA
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    # EMA
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    
    # Volumen
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    
    # Generar señales
    df['signal'] = 0
    
    # Señales basadas en cruces de medias móviles
    df.loc[(df['sma_20'] > df['sma_50']) & (df['sma_20'].shift() <= df['sma_50'].shift()), 'signal'] = 1
    df.loc[(df['sma_20'] < df['sma_50']) & (df['sma_20'].shift() >= df['sma_50'].shift()), 'signal'] = -1
    
    # Señales basadas en RSI
    df.loc[df['rsi'] < 30, 'signal'] = 1
    df.loc[df['rsi'] > 70, 'signal'] = -1
    
    # Señales basadas en Bandas de Bollinger
    df.loc[df['close'] <= df['bb_lower'], 'signal'] = 1
    df.loc[df['close'] >= df['bb_upper'], 'signal'] = -1
    
    # Fuerza de la señal (aleatoria para demostración)
    df['signal_strength'] = np.random.randint(1, 5, size=len(df))
    
    return df

def generate_recommendation(symbol, risk_profile):
    """Genera una recomendación de trading para demostración"""
    # Generar recomendación aleatoria para demostración
    import random
    
    rec_types = ["COMPRAR", "VENDER", "NEUTRAL"]
    weights = [0.4, 0.4, 0.2]  # Más probabilidad de compra/venta que neutral
    
    rec_type = random.choices(rec_types, weights=weights, k=1)[0]
    
    if symbol.startswith("BTC"):
        price = random.uniform(45000, 55000)
    elif symbol.startswith("ETH"):
        price = random.uniform(2800, 3200)
    elif symbol in ["AAPL", "MSFT", "GOOGL"]:
        price = random.uniform(140, 160)
    else:
        price = random.uniform(90, 110)
    
    # Ajustar stop loss y take profit según perfil de riesgo
    if risk_profile == "Conservador":
        sl_pct = 0.05
        tp_pct = 0.10
    elif risk_profile == "Moderado":
        sl_pct = 0.07
        tp_pct = 0.15
    else:  # Agresivo
        sl_pct = 0.10
        tp_pct = 0.20
    
    if rec_type == "COMPRAR":
        confidence = random.randint(60, 95)
        stop_loss = price * (1 - sl_pct)
        take_profit = price * (1 + tp_pct)
        reasons = [
            "Cruce alcista de SMA 20 por encima de SMA 50",
            "RSI saliendo de zona de sobreventa",
            "Precio rebotando desde la banda inferior de Bollinger",
            "MACD cruzando por encima de la línea de señal",
            "Volumen creciente con precio en alza"
        ]
    elif rec_type == "VENDER":
        confidence = random.randint(60, 95)
        stop_loss = price * (1 + sl_pct)
        take_profit = price * (1 - tp_pct)
        reasons = [
            "Cruce bajista de SMA 20 por debajo de SMA 50",
            "RSI en zona de sobrecompra",
            "Precio tocando la banda superior de Bollinger",
            "MACD cruzando por debajo de la línea de señal",
            "Divergencia bajista en RSI"
        ]
    else:  # NEUTRAL
        confidence = random.randint(30, 50)
        stop_loss = None
        take_profit = None
        reasons = [
            "Señales mixtas en los indicadores técnicos",
            "Baja volatilidad en el mercado",
            "Precio en rango lateral",
            "Volumen por debajo del promedio"
        ]
    
    # Seleccionar razones aleatorias
    num_reasons = random.randint(1, 3)
    selected_reasons = random.sample(reasons, num_reasons)
    reason = "; ".join(selected_reasons)
    
    return {
        "symbol": symbol,
        "recommendation": rec_type,
        "confidence": confidence,
        "entry_price": price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "reason": reason,
        "timestamp": datetime.now().isoformat(),
        "all_reasons": reasons
    }

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
            for period in [12, 26]:
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
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">Precio Actual</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">${current_price:.2f}</div>', unsafe_allow_html=True)
        
        # Mostrar cambio con color según dirección
        if price_change >= 0:
            st.markdown(f'<div style="color: green;">+${price_change:.2f} (+{price_change_pct:.2f}%)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="color: red;">${price_change:.2f} ({price_change_pct:.2f}%)</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">Rango de Precios</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">${price_low:.2f} - ${price_high:.2f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div>Últimos {len(df)} períodos</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">Volumen Promedio</div>', unsafe_allow_html=True)
        
        # Formatear volumen para mejor legibilidad
        if avg_volume >= 1_000_000:
            vol_str = f'{avg_volume/1_000_000:.2f}M'
        elif avg_volume >= 1_000:
            vol_str = f'{avg_volume/1_000:.2f}K'
        else:
            vol_str = f'{avg_volume:.2f}'
            
        st.markdown(f'<div class="metric-value">{vol_str}</div>', unsafe_allow_html=True)
        st.markdown(f'<div>Últimos {len(df)} períodos</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">Última Actualización</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{datetime.now().strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div>{datetime.now().strftime("%d/%m/%Y")}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def display_signals_table(df):
    """Muestra una tabla con las señales de trading recientes"""
    if df is None or df.empty:
        return
    
    # Filtrar señales
    signals_df = df[df['signal'] != 0].copy()
    
    if signals_df.empty:
        st.info("No se han detectado señales de trading en el período seleccionado.")
        return
    
    # Limitar a las 10 señales más recientes
    signals_df = signals_df.tail(10)
    
    # Preparar datos para la tabla
    signals_data = []
    
    for _, row in signals_df.iterrows():
        signal_type = "COMPRA" if row['signal'] == 1 else "VENTA"
        signal_strength = row.get('signal_strength', 1)
        
        # Formatear fecha
        date_str = row['timestamp'].strftime('%d/%m/%Y %H:%M')
        
        # Añadir fila a los datos
        signals_data.append({
            "Fecha": date_str,
            "Tipo": signal_type,
            "Precio": f"${row['close']:.2f}",
            "Fuerza": signal_strength
        })
    
    # Mostrar tabla
    st.markdown("### Señales de Trading Recientes")
    st.table(pd.DataFrame(signals_data))

def run_backtest(df, strategy_name):
    """Ejecuta un backtest de la estrategia seleccionada"""
    if df is None or df.empty:
        return None, None
    
    # Copiar dataframe para no modificar el original
    backtest_df = df.copy()
    
    # Aplicar estrategia según selección
    if strategy_name == "sma_crossover":
        # Estrategia de cruce de medias móviles
        backtest_df['signal'] = 0
        
        # Verificar que existan las columnas necesarias
        if all(col in backtest_df.columns for col in ['sma_20', 'sma_50']):
            # Señal de compra: SMA corta cruza por encima de SMA larga
            backtest_df.loc[(backtest_df['sma_20'] > backtest_df['sma_50']) & 
                          (backtest_df['sma_20'].shift() <= backtest_df['sma_50'].shift()), 'signal'] = 1
            
            # Señal de venta: SMA corta cruza por debajo de SMA larga
            backtest_df.loc[(backtest_df['sma_20'] < backtest_df['sma_50']) & 
                          (backtest_df['sma_20'].shift() >= backtest_df['sma_50'].shift()), 'signal'] = -1
    
    elif strategy_name == "rsi_strategy":
        # Estrategia basada en RSI
        backtest_df['signal'] = 0
        
        if 'rsi' in backtest_df.columns:
            # Señal de compra: RSI por debajo de 30
            backtest_df.loc[backtest_df['rsi'] < 30, 'signal'] = 1
            
            # Señal de venta: RSI por encima de 70
            backtest_df.loc[backtest_df['rsi'] > 70, 'signal'] = -1
    
    elif strategy_name == "macd_strategy":
        # Estrategia basada en MACD
        backtest_df['signal'] = 0
        
        if all(col in backtest_df.columns for col in ['macd', 'macd_signal']):
            # Señal de compra: MACD cruza por encima de la línea de señal
            backtest_df.loc[(backtest_df['macd'] > backtest_df['macd_signal']) & 
                          (backtest_df['macd'].shift() <= backtest_df['macd_signal'].shift()), 'signal'] = 1
            
            # Señal de venta: MACD cruza por debajo de la línea de señal
            backtest_df.loc[(backtest_df['macd'] < backtest_df['macd_signal']) & 
                          (backtest_df['macd'].shift() >= backtest_df['macd_signal'].shift()), 'signal'] = -1
    
    elif strategy_name == "bollinger_strategy":
        # Estrategia basada en Bandas de Bollinger
        backtest_df['signal'] = 0
        
        if all(col in backtest_df.columns for col in ['close', 'bb_lower', 'bb_upper']):
            # Señal de compra: Precio toca la banda inferior
            backtest_df.loc[backtest_df['close'] <= backtest_df['bb_lower'], 'signal'] = 1
            
            # Señal de venta: Precio toca la banda superior
            backtest_df.loc[backtest_df['close'] >= backtest_df['bb_upper'], 'signal'] = -1
    
    # Calcular retornos
    backtest_df['position'] = backtest_df['signal'].shift(1).fillna(0)
    backtest_df['returns'] = backtest_df['close'].pct_change()
    backtest_df['strategy_returns'] = backtest_df['position'] * backtest_df['returns']
    backtest_df['cumulative_returns'] = (1 + backtest_df['returns']).cumprod() - 1
    backtest_df['cumulative_strategy_returns'] = (1 + backtest_df['strategy_returns']).cumprod() - 1
    
    # Calcular drawdown
    backtest_df['peak'] = backtest_df['cumulative_strategy_returns'].cummax()
    backtest_df['drawdown'] = (backtest_df['cumulative_strategy_returns'] - backtest_df['peak']) / (1 + backtest_df['peak']) * 100
    
    # Calcular métricas
    metrics = calculate_backtest_metrics(backtest_df)
    
    return backtest_df, metrics

def calculate_backtest_metrics(df):
    """Calcula métricas de rendimiento para el backtest"""
    if df is None or df.empty:
        return None
    
    # Número de operaciones
    trades = len(df[df['signal'] != 0])
    
    # Rendimiento total
    if 'cumulative_strategy_returns' in df.columns and len(df) > 0:
        total_return = df['cumulative_strategy_returns'].iloc[-1] * 100
    else:
        total_return = 0
    
    # Rendimiento de buy & hold
    if 'cumulative_returns' in df.columns and len(df) > 0:
        buy_hold_return = df['cumulative_returns'].iloc[-1] * 100
    else:
        buy_hold_return = 0
    
    # Ratio de operaciones ganadoras
    winning_trades = len(df[df['strategy_returns'] > 0])
    win_ratio = winning_trades / trades if trades > 0 else 0
    
    # Máximo drawdown
    max_drawdown = df['drawdown'].min() if 'drawdown' in df.columns else 0
    
    # Ratio de Sharpe (simplificado)
    if 'strategy_returns' in df.columns and len(df) > 0:
        mean_return = df['strategy_returns'].mean() * 252  # Anualizado
        std_return = df['strategy_returns'].std() * np.sqrt(252)  # Anualizado
        sharpe_ratio = mean_return / std_return if std_return > 0 else 0
    else:
        sharpe_ratio = 0
    
    return {
        'trades': trades,
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'win_ratio': win_ratio,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio
    }

def plot_backtest_results(df, strategy_name):
    """Genera gráficos para visualizar los resultados del backtest"""
    if df is None or df.empty:
        return None, None
    
    # Crear figura
    fig = go.Figure()
    
    # Gráfico de rendimientos acumulados
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['cumulative_strategy_returns'] * 100,
        mode='lines',
        name=f'Estrategia {strategy_name}',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['cumulative_returns'] * 100,
        mode='lines',
        name='Buy & Hold',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    # Configurar diseño
    fig.update_layout(
        title=f'Resultados de Backtest: {strategy_name}',
        xaxis_title='Fecha',
        yaxis_title='Rendimiento (%)',
        height=500,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Crear figura para drawdown
    fig_drawdown = go.Figure()
    
    fig_drawdown.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['drawdown'],
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='red', width=1),
        name='Drawdown'
    ))
    
    fig_drawdown.update_layout(
        title='Drawdown',
        xaxis_title='Fecha',
        yaxis_title='Drawdown (%)',
        height=250,
        template='plotly_white'
    )
    
    return fig, fig_drawdown

# Interfaz principal
def main():
    # Título principal
    st.markdown('<h1 class="main-header">Software de Trading en Futuros</h1>', unsafe_allow_html=True)
    
    # Barra lateral para configuración
    with st.sidebar:
        st.markdown("## Configuración")
        
        # Selección de categoría
        category = st.selectbox(
            "Categoría",
            ["Criptomonedas", "Acciones", "Futuros"]
        )
        
        # Obtener símbolos según categoría
        if category == "Criptomonedas":
            symbols_list = DEFAULT_SYMBOLS["crypto"]
        elif category == "Acciones":
            symbols_list = DEFAULT_SYMBOLS["stocks"]
        else:
            symbols_list = DEFAULT_SYMBOLS["futures"]
        
        # Crear lista para selectbox
        symbols_options = [f"{s['symbol']} - {s['name']}" for s in symbols_list]
        
        # Selección de símbolo
        symbol_option = st.selectbox(
            "Símbolo",
            symbols_options
        )
        
        # Extraer símbolo seleccionado
        symbol = symbol_option.split(" - ")[0]
        
        # Selección de intervalo
        interval_options = [f"{t['value']} - {t['name']}" for t in TIME_INTERVALS]
        interval_option = st.selectbox(
            "Intervalo",
            interval_options,
            index=6  # Por defecto 1d
        )
        interval = interval_option.split(" - ")[0]
        
        # Selección de período
        period_options = [f"{p['value']} - {p['name']}" for p in TIME_PERIODS]
        period_option = st.selectbox(
            "Período",
            period_options,
            index=3  # Por defecto 3mo
        )
        period = period_option.split(" - ")[0]
        
        # Selección de perfil de riesgo
        risk_profile = st.selectbox(
            "Perfil de Riesgo",
            ["Conservador", "Moderado", "Agresivo"],
            index=1  # Por defecto Moderado
        )
        
        # Selección de indicadores
        st.markdown("### Indicadores Técnicos")
        
        indicators = []
        
        if st.checkbox("Medias Móviles (SMA)", value=True):
            indicators.append("sma")
        
        if st.checkbox("Medias Móviles Exponenciales (EMA)"):
            indicators.append("ema")
        
        if st.checkbox("Bandas de Bollinger", value=True):
            indicators.append("bollinger")
        
        # Botón para cargar datos
        if st.button("Cargar Datos"):
            st.session_state.symbol = symbol
            st.session_state.interval = interval
            st.session_state.period = period
            st.session_state.risk_profile = risk_profile
            st.session_state.indicators = indicators
            st.session_state.data_loaded = True
    
    # Inicializar estado de sesión si no existe
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if 'symbol' not in st.session_state:
        st.session_state.symbol = "BTC-USD"
    
    if 'indicators' not in st.session_state:
        st.session_state.indicators = ["sma", "bollinger"]
    
    if 'risk_profile' not in st.session_state:
        st.session_state.risk_profile = "Moderado"
    
    # Contenido principal
    if st.session_state.data_loaded:
        # Cargar datos de ejemplo para demostración
        df = generate_sample_data(st.session_state.symbol)
        
        # Mostrar métricas
        display_metrics(df, st.session_state.symbol)
        
        # Crear pestañas
        tab1, tab2, tab3, tab4 = st.tabs(["Análisis de Mercado", "Señales de Trading", "Recomendaciones", "Backtesting"])
        
        with tab1:
            # Gráfico principal
            st.markdown('<h2 class="sub-header">Análisis Técnico</h2>', unsafe_allow_html=True)
            
            fig = create_candlestick_chart(df, st.session_state.indicators)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Gráficos de indicadores
            st.markdown('<h2 class="sub-header">Indicadores Técnicos</h2>', unsafe_allow_html=True)
            
            indicator_charts = create_indicator_charts(df)
            if indicator_charts:
                for chart in indicator_charts:
                    st.plotly_chart(chart, use_container_width=True)
        
        with tab2:
            st.markdown('<h2 class="sub-header">Señales de Trading</h2>', unsafe_allow_html=True)
            
            # Mostrar tabla de señales
            display_signals_table(df)
            
            # Explicación de señales
            st.markdown("""
            ### Interpretación de Señales
            
            Las señales de trading se generan a partir del análisis de múltiples indicadores técnicos:
            
            - **Señales de COMPRA**: Se generan cuando los indicadores sugieren un posible movimiento alcista.
            - **Señales de VENTA**: Se generan cuando los indicadores sugieren un posible movimiento bajista.
            - **Fuerza de la Señal**: Indica la convicción de la señal basada en la confirmación de múltiples indicadores.
            
            *Nota: Las señales deben ser confirmadas con análisis adicional antes de tomar decisiones de trading.*
            """)
        
        with tab3:
            st.markdown('<h2 class="sub-header">Recomendaciones de Trading</h2>', unsafe_allow_html=True)
            
            # Generar recomendación
            with st.spinner("Generando recomendación..."):
                recommendation = generate_recommendation(st.session_state.symbol, st.session_state.risk_profile)
            
            # Mostrar recomendación
            if recommendation:
                st.markdown(format_recommendation(recommendation), unsafe_allow_html=True)
                
                # Mostrar razones adicionales
                if "all_reasons" in recommendation and recommendation["all_reasons"]:
                    with st.expander("Ver todas las razones"):
                        for reason in recommendation["all_reasons"]:
                            st.markdown(f"- {reason}")
            else:
                st.error("No se pudo generar una recomendación. Por favor, intente con otro símbolo o período.")
            
            # Advertencia
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Advertencia:</strong> Las recomendaciones se basan únicamente en análisis técnico y no consideran factores fundamentales o noticias del mercado. 
                Utilice esta información como una herramienta complementaria y no como único criterio para sus decisiones de inversión.
            </div>
            """, unsafe_allow_html=True)
        
        with tab4:
            st.markdown('<h2 class="sub-header">Backtesting de Estrategias</h2>', unsafe_allow_html=True)
            
            # Selección de estrategia
            strategy_options = {
                "sma_crossover": "Cruce de Medias Móviles",
                "rsi_strategy": "Estrategia RSI",
                "macd_strategy": "Estrategia MACD",
                "bollinger_strategy": "Estrategia Bandas de Bollinger"
            }
            
            strategy = st.selectbox(
                "Seleccione una estrategia",
                list(strategy_options.keys()),
                format_func=lambda x: strategy_options[x]
            )
            
            # Ejecutar backtest
            if st.button("Ejecutar Backtest"):
                with st.spinner("Ejecutando backtest..."):
                    backtest_df, metrics = run_backtest(df, strategy)
                    
                    if backtest_df is not None and metrics is not None:
                        # Mostrar métricas
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Rendimiento Total", f"{metrics['total_return']:.2f}%")
                            st.metric("Rendimiento Buy & Hold", f"{metrics['buy_hold_return']:.2f}%")
                        
                        with col2:
                            st.metric("Operaciones", metrics['trades'])
                            st.metric("Ratio de Ganancia", f"{metrics['win_ratio']:.2f}")
                        
                        with col3:
                            st.metric("Drawdown Máximo", f"{metrics['max_drawdown']:.2f}%")
                            st.metric("Ratio de Sharpe", f"{metrics['sharpe_ratio']:.2f}")
                        
                        # Mostrar gráficos
                        fig, fig_drawdown = plot_backtest_results(backtest_df, strategy_options[strategy])
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        if fig_drawdown:
                            st.plotly_chart(fig_drawdown, use_container_width=True)
                    else:
                        st.error("No se pudo ejecutar el backtest. Verifique que los datos contengan los indicadores necesarios.")
    else:
        # Mensaje inicial
        st.markdown("""
        <div class="info-box">
            <h2>Bienvenido al Software de Trading en Futuros</h2>
            <p>Esta herramienta le permite analizar activos financieros y recibir recomendaciones de trading basadas en análisis técnico.</p>
            <p>Para comenzar, seleccione un activo y configure los parámetros en la barra lateral, luego haga clic en "Cargar Datos".</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Información sobre funcionalidades
        st.markdown("""
        ### Funcionalidades Principales
        
        - **Análisis Técnico**: Visualice gráficos de precios con múltiples indicadores técnicos
        - **Señales de Trading**: Identifique oportunidades de compra y venta
        - **Recomendaciones**: Reciba sugerencias personalizadas según su perfil de riesgo
        - **Backtesting**: Evalúe el rendimiento histórico de diferentes estrategias
        
        ### Activos Disponibles
        
        - **Criptomonedas**: Bitcoin, Ethereum, Solana y más
        - **Acciones**: Apple, Microsoft, Google y otras empresas líderes
        - **Futuros**: S&P 500, Nasdaq, Oro, Petróleo y más
        """)

if __name__ == "__main__":
    main()
