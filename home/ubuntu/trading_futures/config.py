"""
Archivo de configuración para el Software de Trading en Futuros.
Contiene parámetros optimizados para mejorar el rendimiento y la precisión.
"""

# Configuración general
CONFIG = {
    # Proveedores de datos
    "data_providers": {
        "default": "yahoo",
        "available": ["yahoo", "binance"],
        "api_keys": {
            "binance": {
                "api_key": "",  # Añadir clave API aquí si se utiliza
                "api_secret": ""  # Añadir secreto API aquí si se utiliza
            }
        },
        "cache_timeout": 300,  # Tiempo de caché en segundos
        "max_retries": 3,      # Número máximo de reintentos para solicitudes fallidas
    },
    
    # Parámetros de análisis técnico
    "technical_analysis": {
        "default_periods": {
            "sma": [20, 50, 200],
            "ema": [12, 26, 50],
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "rsi": 14,
            "bollinger": {"period": 20, "std_dev": 2.0},
            "stochastic": {"k_period": 14, "d_period": 3},
            "atr": 14,
            "adx": 14
        },
        "signal_threshold": 0.5,  # Umbral para generar señales
        "signal_confirmation": 2,  # Número de confirmaciones requeridas
    },
    
    # Perfiles de riesgo
    "risk_profiles": {
        "conservative": {
            "stop_loss_pct": 5,
            "take_profit_pct": 10,
            "max_trades_per_day": 2,
            "max_portfolio_risk": 15
        },
        "moderate": {
            "stop_loss_pct": 7,
            "take_profit_pct": 15,
            "max_trades_per_day": 5,
            "max_portfolio_risk": 25
        },
        "aggressive": {
            "stop_loss_pct": 10,
            "take_profit_pct": 20,
            "max_trades_per_day": 10,
            "max_portfolio_risk": 40
        }
    },
    
    # Configuración de la interfaz de usuario
    "ui": {
        "theme": "light",  # "light" o "dark"
        "default_chart_type": "candlestick",  # "candlestick", "line", "ohlc"
        "default_interval": "1d",
        "default_period": "3mo",
        "default_symbols": ["BTC-USD", "ETH-USD", "AAPL", "MSFT"],
        "max_indicators_per_chart": 5,
        "refresh_interval": 60,  # Intervalo de actualización en segundos
    },
    
    # Optimización de rendimiento
    "performance": {
        "parallel_requests": True,  # Habilitar solicitudes paralelas
        "max_workers": 4,           # Número máximo de workers para procesamiento paralelo
        "use_cache": True,          # Habilitar caché de datos
        "cache_dir": ".cache",      # Directorio para archivos de caché
        "log_level": "INFO",        # Nivel de logging (DEBUG, INFO, WARNING, ERROR)
    }
}

# Símbolos disponibles por defecto
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

# Indicadores técnicos disponibles
TECHNICAL_INDICATORS = [
    {"id": "sma", "name": "Media Móvil Simple (SMA)", "category": "Tendencia"},
    {"id": "ema", "name": "Media Móvil Exponencial (EMA)", "category": "Tendencia"},
    {"id": "macd", "name": "MACD", "category": "Momentum"},
    {"id": "rsi", "name": "Índice de Fuerza Relativa (RSI)", "category": "Momentum"},
    {"id": "bollinger", "name": "Bandas de Bollinger", "category": "Volatilidad"},
    {"id": "stochastic", "name": "Oscilador Estocástico", "category": "Momentum"},
    {"id": "atr", "name": "Average True Range (ATR)", "category": "Volatilidad"},
    {"id": "adx", "name": "Average Directional Index (ADX)", "category": "Tendencia"},
    {"id": "ichimoku", "name": "Ichimoku Cloud", "category": "Tendencia"},
    {"id": "fibonacci", "name": "Retrocesos de Fibonacci", "category": "Retrocesos"},
    {"id": "volume", "name": "Indicadores de Volumen", "category": "Volumen"},
    {"id": "support_resistance", "name": "Soporte y Resistencia", "category": "Niveles de Precio"}
]

# Estrategias de trading disponibles
TRADING_STRATEGIES = [
    {"id": "sma_crossover", "name": "Cruce de Medias Móviles", "description": "Compra cuando SMA corta cruza por encima de SMA larga, vende en caso contrario"},
    {"id": "rsi_strategy", "name": "Estrategia RSI", "description": "Compra cuando RSI < 30, vende cuando RSI > 70"},
    {"id": "macd_strategy", "name": "Estrategia MACD", "description": "Compra cuando MACD cruza por encima de la línea de señal, vende en caso contrario"},
    {"id": "bollinger_strategy", "name": "Estrategia Bandas de Bollinger", "description": "Compra cuando el precio toca la banda inferior, vende cuando toca la banda superior"},
    {"id": "multi_indicator", "name": "Estrategia Multi-Indicador", "description": "Combina señales de múltiples indicadores con un sistema de puntuación ponderada"}
]
