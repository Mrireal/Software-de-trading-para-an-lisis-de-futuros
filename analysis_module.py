"""
Módulo de Análisis Técnico para el Software de Trading en Futuros

Este módulo implementa diversos algoritmos de análisis técnico para identificar
patrones y generar señales de trading basadas en datos históricos de precios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("analysis_module")

class TechnicalAnalysis:
    """Clase principal para realizar análisis técnico en datos financieros"""
    
    def __init__(self):
        """Inicializa el analizador técnico"""
        logger.info("Inicializando módulo de análisis técnico")
    
    def add_moving_averages(self, df, windows=[20, 50, 200]):
        """
        Añade medias móviles simples (SMA) al DataFrame
        
        Args:
            df (pandas.DataFrame): DataFrame con datos financieros
            windows (list): Lista de períodos para las medias móviles
            
        Returns:
            pandas.DataFrame: DataFrame con medias móviles añadidas
        """
        if df is None or df.empty or 'close' not in df.columns:
            logger.warning("DataFrame inválido para calcular medias móviles")
            return df
        
        df_result = df.copy()
        
        for window in windows:
            df_result[f'sma_{window}'] = df_result['close'].rolling(window=window).mean()
            logger.info(f"Media móvil SMA-{window} calculada")
        
        return df_result
    
    def add_exponential_moving_averages(self, df, windows=[12, 26, 50]):
        """
        Añade medias móviles exponenciales (EMA) al DataFrame
        
        Args:
            df (pandas.DataFrame): DataFrame con datos financieros
            windows (list): Lista de períodos para las medias móviles
            
        Returns:
            pandas.DataFrame: DataFrame con EMAs añadidas
        """
        if df is None or df.empty or 'close' not in df.columns:
            logger.warning("DataFrame inválido para calcular EMAs")
            return df
        
        df_result = df.copy()
        
        for window in windows:
            df_result[f'ema_{window}'] = df_result['close'].ewm(span=window, adjust=False).mean()
            logger.info(f"Media móvil EMA-{window} calculada")
        
        return df_result
    
    def add_macd(self, df, fast_period=12, slow_period=26, signal_period=9):
        """
        Añade el indicador MACD (Moving Average Convergence Divergence) al DataFrame
        
        Args:
            df (pandas.DataFrame): DataFrame con datos financieros
            fast_period (int): Período para la EMA rápida
            slow_period (int): Período para la EMA lenta
            signal_period (int): Período para la línea de señal
            
        Returns:
            pandas.DataFrame: DataFrame con MACD añadido
        """
        if df is None or df.empty or 'close' not in df.columns:
            logger.warning("DataFrame inválido para calcular MACD")
            return df
        
        df_result = df.copy()
        
        # Calcular EMAs
        ema_fast = df_result['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df_result['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calcular MACD y señal
        df_result['macd'] = ema_fast - ema_slow
        df_result['macd_signal'] = df_result['macd'].ewm(span=signal_period, adjust=False).mean()
        df_result['macd_histogram'] = df_result['macd'] - df_result['macd_signal']
        
        logger.info(f"MACD calculado con períodos {fast_period}/{slow_period}/{signal_period}")
        
        return df_result
    
    def add_rsi(self, df, period=14):
        """
        Añade el indicador RSI (Relative Strength Index) al DataFrame
        
        Args:
            df (pandas.DataFrame): DataFrame con datos financieros
            period (int): Período para el cálculo del RSI
            
        Returns:
            pandas.DataFrame: DataFrame con RSI añadido
        """
        if df is None or df.empty or 'close' not in df.columns:
            logger.warning("DataFrame inválido para calcular RSI")
            return df
        
        df_result = df.copy()
        
        # Calcular cambios en el precio
        delta = df_result['close'].diff()
        
        # Separar ganancias (up) y pérdidas (down)
        up = delta.copy()
        up[up < 0] = 0
        down = -1 * delta.copy()
        down[down < 0] = 0
        
        # Calcular la media móvil exponencial de ganancias y pérdidas
        avg_gain = up.ewm(com=period-1, min_periods=period).mean()
        avg_loss = down.ewm(com=period-1, min_periods=period).mean()
        
        # Calcular RS y RSI
        rs = avg_gain / avg_loss
        df_result['rsi'] = 100 - (100 / (1 + rs))
        
        logger.info(f"RSI calculado con período {period}")
        
        return df_result
    
    def add_bollinger_bands(self, df, period=20, std_dev=2):
        """
        Añade Bandas de Bollinger al DataFrame
        
        Args:
            df (pandas.DataFrame): DataFrame con datos financieros
            period (int): Período para la media móvil
            std_dev (float): Número de desviaciones estándar
            
        Returns:
            pandas.DataFrame: DataFrame con Bandas de Bollinger añadidas
        """
        if df is None or df.empty or 'close' not in df.columns:
            logger.warning("DataFrame inválido para calcular Bandas de Bollinger")
            return df
        
        df_result = df.copy()
        
        # Calcular media móvil
        df_result[f'bb_middle'] = df_result['close'].rolling(window=period).mean()
        
        # Calcular desviación estándar
        rolling_std = df_result['close'].rolling(window=period).std()
        
        # Calcular bandas superior e inferior
        df_result['bb_upper'] = df_result['bb_middle'] + (rolling_std * std_dev)
        df_result['bb_lower'] = df_result['bb_middle'] - (rolling_std * std_dev)
        
        # Calcular ancho de banda (indicador de volatilidad)
        df_result['bb_width'] = (df_result['bb_upper'] - df_result['bb_lower']) / df_result['bb_middle']
        
        logger.info(f"Bandas de Bollinger calculadas con período {period} y {std_dev} desviaciones estándar")
        
        return df_result
    
    def add_stochastic_oscillator(self, df, k_period=14, d_period=3):
        """
        Añade el Oscilador Estocástico al DataFrame
        
        Args:
            df (pandas.DataFrame): DataFrame con datos financieros
            k_period (int): Período para %K
            d_period (int): Período para %D
            
        Returns:
            pandas.DataFrame: DataFrame con Oscilador Estocástico añadido
        """
        if df is None or df.empty or not all(col in df.columns for col in ['high', 'low', 'close']):
            logger.warning("DataFrame inválido para calcular Oscilador Estocástico")
            return df
        
        df_result = df.copy()
        
        # Calcular %K
        low_min = df_result['low'].rolling(window=k_period).min()
        high_max = df_result['high'].rolling(window=k_period).max()
        
        df_result['stoch_k'] = 100 * ((df_result['close'] - low_min) / (high_max - low_min))
        
        # Calcular %D (media móvil de %K)
        df_result['stoch_d'] = df_result['stoch_k'].rolling(window=d_period).mean()
        
        logger.info(f"Oscilador Estocástico calculado con períodos K={k_period}, D={d_period}")
        
        return df_result
    
    def add_atr(self, df, period=14):
        """
        Añade el indicador ATR (Average True Range) al DataFrame
        
        Args:
            df (pandas.DataFrame): DataFrame con datos financieros
            period (int): Período para el cálculo del ATR
            
        Returns:
            pandas.DataFrame: DataFrame con ATR añadido
        """
        if df is None or df.empty or not all(col in df.columns for col in ['high', 'low', 'close']):
            logger.warning("DataFrame inválido para calcular ATR")
            return df
        
        df_result = df.copy()
        
        # Calcular True Range
        df_result['tr0'] = abs(df_result['high'] - df_result['low'])
        df_result['tr1'] = abs(df_result['high'] - df_result['close'].shift())
        df_result['tr2'] = abs(df_result['low'] - df_result['close'].shift())
        df_result['tr'] = df_result[['tr0', 'tr1', 'tr2']].max(axis=1)
        
        # Calcular ATR (media móvil exponencial del True Range)
        df_result['atr'] = df_result['tr'].ewm(span=period, adjust=False).mean()
        
        # Eliminar columnas temporales
        df_result = df_result.drop(['tr0', 'tr1', 'tr2', 'tr'], axis=1)
        
        logger.info(f"ATR calculado con período {period}")
        
        return df_result
    
    def add_adx(self, df, period=14):
        """
        Añade el indicador ADX (Average Directional Index) al DataFrame
        
        Args:
            df (pandas.DataFrame): DataFrame con datos financieros
            period (int): Período para el cálculo del ADX
            
        Returns:
            pandas.DataFrame: DataFrame con ADX añadido
        """
        if df is None or df.empty or not all(col in df.columns for col in ['high', 'low', 'close']):
            logger.warning("DataFrame inválido para calcular ADX")
            return df
        
        df_result = df.copy()
        
        # Calcular +DM y -DM
        df_result['up_move'] = df_result['high'].diff()
        df_result['down_move'] = df_result['low'].shift().diff(-1).abs()
        
        df_result['plus_dm'] = np.where(
            (df_result['up_move'] > df_result['down_move']) & (df_result['up_move'] > 0),
            df_result['up_move'],
            0
        )
        
        df_result['minus_dm'] = np.where(
            (df_result['down_move'] > df_result['up_move']) & (df_result['down_move'] > 0),
            df_result['down_move'],
            0
        )
        
        # Calcular ATR
        df_result = self.add_atr(df_result, period)
        
        # Calcular +DI y -DI
        df_result['plus_di'] = 100 * (df_result['plus_dm'].ewm(span=period, adjust=False).mean() / df_result['atr'])
        df_result['minus_di'] = 100 * (df_result['minus_dm'].ewm(span=period, adjust=False).mean() / df_result['atr'])
        
        # Calcular DX y ADX
        df_result['dx'] = 100 * abs(df_result['plus_di'] - df_result['minus_di']) / (df_result['plus_di'] + df_result['minus_di'])
        df_result['adx'] = df_result['dx'].ewm(span=period, adjust=False).mean()
        
        # Eliminar columnas temporales
        df_result = df_result.drop(['up_move', 'down_move', 'plus_dm', 'minus_dm'], axis=1)
        
        logger.info(f"ADX calculado con período {period}")
        
        return df_result
    
    def add_ichimoku_cloud(self, df, tenkan_period=9, kijun_period=26, senkou_b_period=52, displacement=26):
        """
        Añade el indicador Ichimoku Cloud al DataFrame
        
        Args:
            df (pandas.DataFrame): DataFrame con datos financieros
            tenkan_period (int): Período para Tenkan-sen (línea de conversión)
            kijun_period (int): Período para Kijun-sen (línea base)
            senkou_b_period (int): Período para Senkou Span B
            displacement (int): Período de desplazamiento para Senkou Span
            
        Returns:
            pandas.DataFrame: DataFrame con Ichimoku Cloud añadido
        """
        if df is None or df.empty or not all(col in df.columns for col in ['high', 'low', 'close']):
            logger.warning("DataFrame inválido para calcular Ichimoku Cloud")
            return df
        
        df_result = df.copy()
        
        # Tenkan-sen (línea de conversión)
        tenkan_high = df_result['high'].rolling(window=tenkan_period).max()
        tenkan_low = df_result['low'].rolling(window=tenkan_period).min()
        df_result['ichimoku_tenkan'] = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (línea base)
        kijun_high = df_result['high'].rolling(window=kijun_period).max()
        kijun_low = df_result['low'].rolling(window=kijun_period).min()
        df_result['ichimoku_kijun'] = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (primera línea de la nube)
        df_result['ichimoku_senkou_a'] = ((df_result['ichimoku_tenkan'] + df_result['ichimoku_kijun']) / 2).shift(displacement)
        
        # Senkou Span B (segunda línea de la nube)
        senkou_b_high = df_result['high'].rolling(window=senkou_b_period).max()
        senkou_b_low = df_result['low'].rolling(window=senkou_b_period).min()
        df_result['ichimoku_senkou_b'] = ((senkou_b_high + senkou_b_low) / 2).shift(displacement)
        
        # Chikou Span (línea de retraso)
        df_result['ichimoku_chikou'] = df_result['close'].shift(-displacement)
        
        logger.info(f"Ichimoku Cloud calculado con períodos {tenkan_period}/{kijun_period}/{senkou_b_period}")
        
        return df_result
    
    def add_fibonacci_levels(self, df, period=100):
        """
        Añade niveles de retroceso de Fibonacci basados en máximos y mínimos recientes
        
        Args:
            df (pandas.DataFrame): DataFrame con datos financieros
            period (int): Período para identificar máximos y mínimos
            
        Returns:
            pandas.DataFrame: DataFrame con niveles de Fibonacci añadidos
        """
        if df is None or df.empty or not all(col in df.columns for col in ['high', 'low']):
            logger.warning("DataFrame inválido para calcular niveles de Fibonacci")
            return df
        
        df_result = df.copy()
        
        # Identificar máximo y mínimo en el período
        high = df_result['high'].rolling(window=period).max()
        low = df_result['low'].rolling(window=period).min()
        
        # Calcular niveles de Fibonacci
        diff = high - low
        df_result['fib_0'] = low
        df_result['fib_0.236'] = low + 0.236 * diff
        df_result['fib_0.382'] = low + 0.382 * diff
        df_result['fib_0.5'] = low + 0.5 * diff
        df_result['fib_0.618'] = low + 0.618 * diff
        df_result['fib_0.786'] = low + 0.786 * diff
        df_result['fib_1'] = high
        
        logger.info(f"Niveles de Fibonacci calculados con período {period}")
        
        return df_result
    
    def add_support_resistance(self, df, window=20, sensitivity=3):
        """
        Identifica niveles de soporte y resistencia
        
        Args:
            df (pandas.DataFrame): DataFrame con datos financieros
            window (int): Tamaño de la ventana para buscar máximos y mínimos locales
            sensitivity (int): Sensibilidad para identificar niveles (mayor = menos sensible)
            
        Returns:
            pandas.DataFrame: DataFrame con niveles de soporte y resistencia
        """
        if df is None or df.empty or not all(col in df.columns for col in ['high', 'low']):
            logger.warning("DataFrame inválido para calcular soporte y resistencia")
            return df
        
        df_result = df.copy()
        
        # Inicializar columnas
        df_result['support'] = np.nan
        df_result['resistance'] = np.nan
        
        # Identificar máximos y mínimos locales
        for i in range(window, len(df_result) - window):
            # Verificar si es un mínimo local
            if all(df_result['low'].iloc[i] <= df_result['low'].iloc[i-j] for j in range(1, sensitivity+1)) and \
               all(df_result['low'].iloc[i] <= df_result['low'].iloc[i+j] for j in range(1, s
(Content truncated due to size limit. Use line ranges to read in chunks)