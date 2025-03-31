"""
Utilidades para el módulo de datos del software de trading en futuros.

Este módulo proporciona funciones auxiliares para el procesamiento y manipulación
de datos financieros, incluyendo normalización, transformación y validación.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configuración de logging
logger = logging.getLogger("data_utils")

def validate_dataframe(df):
    """
    Valida que un DataFrame tenga la estructura correcta para datos financieros
    
    Args:
        df (pandas.DataFrame): DataFrame a validar
        
    Returns:
        bool: True si el DataFrame es válido, False en caso contrario
    """
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    # Verificar que el DataFrame no esté vacío
    if df is None or df.empty:
        logger.warning("DataFrame vacío o nulo")
        return False
    
    # Verificar que tenga las columnas requeridas
    for col in required_columns:
        if col not in df.columns:
            logger.warning(f"Columna requerida '{col}' no encontrada en el DataFrame")
            return False
    
    # Verificar que no haya valores nulos en columnas críticas
    critical_columns = ['timestamp', 'close']
    for col in critical_columns:
        if df[col].isnull().any():
            logger.warning(f"Valores nulos encontrados en la columna '{col}'")
            return False
    
    return True

def clean_dataframe(df):
    """
    Limpia un DataFrame de datos financieros
    
    Args:
        df (pandas.DataFrame): DataFrame a limpiar
        
    Returns:
        pandas.DataFrame: DataFrame limpio
    """
    if df is None or df.empty:
        return df
    
    # Crear una copia para no modificar el original
    df_clean = df.copy()
    
    # Asegurar que timestamp sea datetime
    if 'timestamp' in df_clean.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_clean['timestamp']):
            try:
                df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
            except Exception as e:
                logger.error(f"Error al convertir timestamp a datetime: {str(e)}")
    
    # Convertir columnas numéricas
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Eliminar filas con valores nulos en columnas críticas
    critical_columns = ['timestamp', 'close']
    df_clean = df_clean.dropna(subset=critical_columns)
    
    # Ordenar por timestamp
    if 'timestamp' in df_clean.columns:
        df_clean = df_clean.sort_values('timestamp')
    
    # Eliminar duplicados
    if 'timestamp' in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=['timestamp'])
    
    return df_clean

def resample_dataframe(df, interval):
    """
    Cambia la frecuencia de muestreo de un DataFrame de datos financieros
    
    Args:
        df (pandas.DataFrame): DataFrame a remuestrear
        interval (str): Nuevo intervalo ('1min', '5min', '15min', '30min', '1h', '4h', '1d', '1w', '1M')
        
    Returns:
        pandas.DataFrame: DataFrame remuestreado
    """
    if df is None or df.empty or 'timestamp' not in df.columns:
        return df
    
    # Crear una copia para no modificar el original
    df_resampled = df.copy()
    
    # Asegurar que timestamp sea datetime
    if not pd.api.types.is_datetime64_any_dtype(df_resampled['timestamp']):
        df_resampled['timestamp'] = pd.to_datetime(df_resampled['timestamp'])
    
    # Establecer timestamp como índice
    df_resampled = df_resampled.set_index('timestamp')
    
    # Mapear el intervalo al formato de pandas
    interval_map = {
        '1min': '1T', '5min': '5T', '15min': '15T', '30min': '30T',
        '1h': '1H', '4h': '4H', '1d': '1D', '1w': '1W', '1M': '1M'
    }
    
    pandas_interval = interval_map.get(interval, '1D')  # Por defecto 1 día
    
    # Remuestrear
    resampled = df_resampled.resample(pandas_interval).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # Restablecer el índice
    resampled = resampled.reset_index()
    
    return resampled

def calculate_returns(df, period=1):
    """
    Calcula los retornos para un DataFrame de datos financieros
    
    Args:
        df (pandas.DataFrame): DataFrame con datos financieros
        period (int): Período para calcular los retornos
        
    Returns:
        pandas.DataFrame: DataFrame con columnas adicionales de retornos
    """
    if df is None or df.empty or 'close' not in df.columns:
        return df
    
    # Crear una copia para no modificar el original
    df_returns = df.copy()
    
    # Calcular retornos simples
    df_returns[f'return_{period}'] = df_returns['close'].pct_change(period)
    
    # Calcular retornos logarítmicos
    df_returns[f'log_return_{period}'] = np.log(df_returns['close'] / df_returns['close'].shift(period))
    
    return df_returns

def normalize_dataframe(df, method='minmax'):
    """
    Normaliza las columnas numéricas de un DataFrame de datos financieros
    
    Args:
        df (pandas.DataFrame): DataFrame a normalizar
        method (str): Método de normalización ('minmax', 'zscore')
        
    Returns:
        pandas.DataFrame: DataFrame normalizado
    """
    if df is None or df.empty:
        return df
    
    # Crear una copia para no modificar el original
    df_norm = df.copy()
    
    # Columnas a normalizar
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    columns_to_normalize = [col for col in numeric_columns if col in df_norm.columns]
    
    if method == 'minmax':
        # Normalización Min-Max (valores entre 0 y 1)
        for col in columns_to_normalize:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val > min_val:  # Evitar división por cero
                df_norm[f'{col}_norm'] = (df_norm[col] - min_val) / (max_val - min_val)
            else:
                df_norm[f'{col}_norm'] = 0
    
    elif method == 'zscore':
        # Normalización Z-Score (media 0, desviación estándar 1)
        for col in columns_to_normalize:
            mean_val = df_norm[col].mean()
            std_val = df_norm[col].std()
            if std_val > 0:  # Evitar división por cero
                df_norm[f'{col}_norm'] = (df_norm[col] - mean_val) / std_val
            else:
                df_norm[f'{col}_norm'] = 0
    
    return df_norm

def add_date_features(df):
    """
    Añade características basadas en la fecha a un DataFrame de datos financieros
    
    Args:
        df (pandas.DataFrame): DataFrame con datos financieros
        
    Returns:
        pandas.DataFrame: DataFrame con características adicionales
    """
    if df is None or df.empty or 'timestamp' not in df.columns:
        return df
    
    # Crear una copia para no modificar el original
    df_features = df.copy()
    
    # Asegurar que timestamp sea datetime
    if not pd.api.types.is_datetime64_any_dtype(df_features['timestamp']):
        df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
    
    # Extraer características de fecha
    df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
    df_features['day_of_month'] = df_features['timestamp'].dt.day
    df_features['month'] = df_features['timestamp'].dt.month
    df_features['year'] = df_features['timestamp'].dt.year
    df_features['hour'] = df_features['timestamp'].dt.hour
    
    # Características cíclicas para día de la semana (valores entre -1 y 1)
    df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
    
    # Características cíclicas para mes (valores entre -1 y 1)
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    
    # Características cíclicas para hora (valores entre -1 y 1)
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
    
    return df_features

def calculate_volatility(df, window=20):
    """
    Calcula la volatilidad para un DataFrame de datos financieros
    
    Args:
        df (pandas.DataFrame): DataFrame con datos financieros
        window (int): Ventana para calcular la volatilidad
        
    Returns:
        pandas.DataFrame: DataFrame con columna adicional de volatilidad
    """
    if df is None or df.empty or 'close' not in df.columns:
        return df
    
    # Crear una copia para no modificar el original
    df_vol = df.copy()
    
    # Calcular retornos logarítmicos
    df_vol['log_return'] = np.log(df_vol['close'] / df_vol['close'].shift(1))
    
    # Calcular volatilidad como desviación estándar móvil de los retornos
    df_vol[f'volatility_{window}'] = df_vol['log_return'].rolling(window=window).std() * np.sqrt(252)
    
    # Eliminar la columna temporal de retornos logarítmicos
    df_vol = df_vol.drop(columns=['log_return'])
    
    return df_vol

def merge_dataframes(df_list, on='timestamp', how='inner'):
    """
    Combina múltiples DataFrames de datos financieros
    
    Args:
        df_list (list): Lista de DataFrames a combinar
        on (str): Columna para realizar la unión
        how (str): Tipo de unión ('inner', 'outer', 'left', 'right')
        
    Returns:
        pandas.DataFrame: DataFrame combinado
    """
    if not df_list or len(df_list) == 0:
        return pd.DataFrame()
    
    # Filtrar DataFrames vacíos
    valid_dfs = [df for df in df_list if df is not None and not df.empty and on in df.columns]
    
    if not valid_dfs:
        return pd.DataFrame()
    
    if len(valid_dfs) == 1:
        return valid_dfs[0]
    
    # Realizar la unión de todos los DataFrames
    result = valid_dfs[0]
    for i, df in enumerate(valid_dfs[1:], 1):
        # Añadir sufijos para evitar conflictos de nombres de columnas
        result = pd.merge(result, df, on=on, how=how, suffixes=(f'', f'_{i}'))
    
    return result

def get_market_hours(symbol, date=None):
    """
    Obtiene las horas de mercado para un símbolo específico
    
    Args:
        symbol (str): Símbolo del activo
        date (datetime, optional): Fecha para la que se quieren las horas de mercado
        
    Returns:
        tuple: (hora_apertura, hora_cierre) en UTC
    """
    if date is None:
        date = datetime.now()
    
    # Por defecto, asumimos mercado 24/7 (criptomonedas)
    market_open = datetime(date.year, date.month, date.day, 0, 0, 0)
    market_close = datetime(date.year, date.month, date.day, 23, 59, 59)
    
    # Verificar si es un símbolo de acciones (horario de mercado estándar)
    if '-' not in symbol and symbol.isalpha():
        # Horario de mercado de EE. UU. (9:30 AM - 4:00 PM EST)
        market_open = datetime(date.year, date.month, date.day, 14, 30, 0)  # 9:30 AM EST en UTC
        market_close = datetime(date.year, date.month, date.day, 21, 0, 0)  # 4:00 PM EST en UTC
        
        # Verificar si es fin de semana
        if date.weekday() >= 5:  # 5 = Sábado, 6 = Domingo
            return None, None
    
    return market_open, market_close

def is_market_open(symbol, timestamp=None):
    """
    Verifica si el mercado está abierto para un símbolo específico en un momento dado
    
    Args:
        symbol (str): Símbolo del activo
        timestamp (datetime, optional): Momento para verificar
        
    Returns:
        bool: True si el mercado está abierto, False en caso contrario
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    market_open, market_close = get_market_hours(symbol, timestamp)
    
    # Si no hay horas de mercado (fin de semana para acciones), el mercado está cerrado
    if market_open is None or market_close is None:
        return False
    
    # Verificar si el timestamp está dentro de las horas de mercado
    return market_open <= timestamp <= market_close
