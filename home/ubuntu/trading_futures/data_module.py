"""
Módulo de Obtención de Datos para el Software de Trading en Futuros

Este módulo implementa la capa de datos del software, proporcionando interfaces
para obtener datos financieros de diferentes fuentes como Yahoo Finance, Binance,
y otras APIs de criptomonedas y futuros.
"""

import os
import json
import time
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_module")

class DataProvider(ABC):
    """Clase base abstracta para todos los proveedores de datos"""
    
    @abstractmethod
    def get_historical_data(self, symbol, interval, start_time, end_time=None):
        """Obtiene datos históricos para un símbolo específico"""
        pass
    
    @abstractmethod
    def get_current_price(self, symbol):
        """Obtiene el precio actual para un símbolo específico"""
        pass
    
    @abstractmethod
    def get_symbols_list(self, asset_type=None):
        """Obtiene la lista de símbolos disponibles"""
        pass

class YahooFinanceProvider(DataProvider):
    """Proveedor de datos usando la API de Yahoo Finance"""
    
    def __init__(self):
        self.base_url = "https://pro-api.coinmarketcap.com/v1/"
        self.data_api_path = "/opt/.manus/.sandbox-runtime"
        logger.info("Inicializando proveedor de datos Yahoo Finance")
    
    def get_historical_data(self, symbol, interval='1d', start_time=None, end_time=None):
        """
        Obtiene datos históricos de Yahoo Finance
        
        Args:
            symbol (str): Símbolo del activo (ej. 'BTC-USD')
            interval (str): Intervalo de tiempo ('1m', '5m', '15m', '30m', '60m', '1d', '1wk', '1mo')
            start_time (str/datetime): Tiempo de inicio
            end_time (str/datetime): Tiempo de fin (opcional, por defecto es ahora)
            
        Returns:
            pandas.DataFrame: Datos históricos con columnas [timestamp, open, high, low, close, volume]
        """
        try:
            import sys
            sys.path.append(self.data_api_path)
            from data_api import ApiClient
            
            client = ApiClient()
            
            # Configurar parámetros
            params = {
                'symbol': symbol,
                'interval': interval,
                'range': '1mo'  # Por defecto, se puede cambiar según start_time
            }
            
            # Ajustar el rango según el tiempo de inicio
            if start_time:
                if isinstance(start_time, str):
                    start_time = datetime.strptime(start_time, "%Y-%m-%d")
                
                days_diff = (datetime.now() - start_time).days
                
                if days_diff <= 5:
                    params['range'] = '5d'
                elif days_diff <= 30:
                    params['range'] = '1mo'
                elif days_diff <= 90:
                    params['range'] = '3mo'
                elif days_diff <= 180:
                    params['range'] = '6mo'
                elif days_diff <= 365:
                    params['range'] = '1y'
                elif days_diff <= 730:
                    params['range'] = '2y'
                elif days_diff <= 1825:
                    params['range'] = '5y'
                else:
                    params['range'] = 'max'
            
            # Llamar a la API
            result = client.call_api('YahooFinance/get_stock_chart', query=params)
            
            # Procesar los datos
            if not result or 'chart' not in result or 'result' not in result['chart'] or not result['chart']['result']:
                logger.error(f"No se pudieron obtener datos para {symbol}")
                return pd.DataFrame()
            
            chart_data = result['chart']['result'][0]
            
            # Extraer timestamps y datos de precios
            timestamps = chart_data['timestamp']
            quote_data = chart_data['indicators']['quote'][0]
            
            # Crear DataFrame
            df = pd.DataFrame({
                'timestamp': [datetime.fromtimestamp(ts) for ts in timestamps],
                'open': quote_data['open'],
                'high': quote_data['high'],
                'low': quote_data['low'],
                'close': quote_data['close'],
                'volume': quote_data['volume']
            })
            
            # Limpiar datos NaN
            df = df.dropna()
            
            logger.info(f"Obtenidos {len(df)} registros históricos para {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error al obtener datos históricos de Yahoo Finance: {str(e)}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol):
        """
        Obtiene el precio actual para un símbolo específico
        
        Args:
            symbol (str): Símbolo del activo (ej. 'BTC-USD')
            
        Returns:
            float: Precio actual
        """
        try:
            # Obtenemos los datos más recientes (último día)
            df = self.get_historical_data(symbol, interval='1d', start_time=(datetime.now() - timedelta(days=1)))
            
            if df.empty:
                logger.error(f"No se pudo obtener el precio actual para {symbol}")
                return None
            
            # Devolvemos el último precio de cierre
            return df['close'].iloc[-1]
            
        except Exception as e:
            logger.error(f"Error al obtener precio actual de Yahoo Finance: {str(e)}")
            return None
    
    def get_symbols_list(self, asset_type=None):
        """
        Obtiene una lista de símbolos disponibles
        
        Args:
            asset_type (str, optional): Tipo de activo ('crypto', 'stock', 'futures')
            
        Returns:
            list: Lista de símbolos disponibles
        """
        # Esta función es limitada ya que Yahoo Finance no proporciona una API directa para listar símbolos
        # Devolvemos una lista predefinida de criptomonedas populares
        crypto_symbols = [
            'BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 
            'DOGE-USD', 'AVAX-USD', 'DOT-USD', 'MATIC-USD'
        ]
        
        stock_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT'
        ]
        
        futures_symbols = [
            'ES=F', 'NQ=F', 'YM=F', 'RTY=F', 'ZB=F', 'ZN=F', 'ZF=F', 'ZT=F', 'GC=F', 'SI=F'
        ]
        
        if asset_type == 'crypto':
            return crypto_symbols
        elif asset_type == 'stock':
            return stock_symbols
        elif asset_type == 'futures':
            return futures_symbols
        else:
            return crypto_symbols + stock_symbols + futures_symbols

class BinanceProvider(DataProvider):
    """Proveedor de datos usando la API de Binance"""
    
    def __init__(self, api_key=None, api_secret=None):
        self.base_url = "https://api.binance.com/api/v3"
        self.futures_url = "https://fapi.binance.com/fapi/v1"
        self.api_key = api_key
        self.api_secret = api_secret
        logger.info("Inicializando proveedor de datos Binance")
    
    def get_historical_data(self, symbol, interval='1d', start_time=None, end_time=None):
        """
        Obtiene datos históricos de Binance
        
        Args:
            symbol (str): Símbolo del activo (ej. 'BTCUSDT')
            interval (str): Intervalo de tiempo ('1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M')
            start_time (str/datetime/int): Tiempo de inicio (timestamp en ms o datetime)
            end_time (str/datetime/int): Tiempo de fin (timestamp en ms o datetime)
            
        Returns:
            pandas.DataFrame: Datos históricos con columnas [timestamp, open, high, low, close, volume]
        """
        try:
            # Determinar si es un símbolo de futuros o spot
            is_futures = symbol.endswith('_PERP') or '_USDT_' in symbol
            base_url = self.futures_url if is_futures else self.base_url
            
            # Preparar el símbolo para la API
            api_symbol = symbol.replace('_PERP', '') if is_futures else symbol
            
            # Endpoint para datos de klines (velas)
            endpoint = "/klines" if is_futures else "/klines"
            url = f"{base_url}{endpoint}"
            
            # Preparar parámetros
            params = {
                'symbol': api_symbol,
                'interval': interval,
                'limit': 1000  # Máximo permitido
            }
            
            # Convertir start_time y end_time a timestamp en ms si es necesario
            if start_time:
                if isinstance(start_time, datetime):
                    start_time = int(start_time.timestamp() * 1000)
                elif isinstance(start_time, str):
                    start_time = int(datetime.strptime(start_time, "%Y-%m-%d").timestamp() * 1000)
                params['startTime'] = start_time
                
            if end_time:
                if isinstance(end_time, datetime):
                    end_time = int(end_time.timestamp() * 1000)
                elif isinstance(end_time, str):
                    end_time = int(datetime.strptime(end_time, "%Y-%m-%d").timestamp() * 1000)
                params['endTime'] = end_time
            
            # Realizar la solicitud
            headers = {}
            if self.api_key:
                headers['X-MBX-APIKEY'] = self.api_key
                
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            # Procesar la respuesta
            data = response.json()
            
            # Crear DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convertir tipos de datos
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            # Seleccionar solo las columnas relevantes
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"Obtenidos {len(df)} registros históricos para {symbol} desde Binance")
            return df
            
        except Exception as e:
            logger.error(f"Error al obtener datos históricos de Binance: {str(e)}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol):
        """
        Obtiene el precio actual para un símbolo específico
        
        Args:
            symbol (str): Símbolo del activo (ej. 'BTCUSDT')
            
        Returns:
            float: Precio actual
        """
        try:
            # Determinar si es un símbolo de futuros o spot
            is_futures = symbol.endswith('_PERP') or '_USDT_' in symbol
            base_url = self.futures_url if is_futures else self.base_url
            
            # Preparar el símbolo para la API
            api_symbol = symbol.replace('_PERP', '') if is_futures else symbol
            
            # Endpoint para precio actual
            endpoint = "/ticker/price"
            url = f"{base_url}{endpoint}"
            
            # Parámetros
            params = {'symbol': api_symbol}
            
            # Realizar la solicitud
            headers = {}
            if self.api_key:
                headers['X-MBX-APIKEY'] = self.api_key
                
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            # Procesar la respuesta
            data = response.json()
            
            price = float(data['price'])
            logger.info(f"Precio actual de {symbol}: {price}")
            return price
            
        except Exception as e:
            logger.error(f"Error al obtener precio actual de Binance: {str(e)}")
            return None
    
    def get_symbols_list(self, asset_type=None):
        """
        Obtiene una lista de símbolos disponibles en Binance
        
        Args:
            asset_type (str, optional): Tipo de activo ('spot', 'futures', 'all')
            
        Returns:
            list: Lista de símbolos disponibles
        """
        try:
            symbols = []
            
            # Obtener símbolos de spot si es necesario
            if asset_type in ['spot', 'all', None]:
                spot_url = f"{self.base_url}/exchangeInfo"
                spot_response = requests.get(spot_url)
                spot_response.raise_for_status()
                spot_data = spot_response.json()
                
                spot_symbols = [item['symbol'] for item in spot_data['symbols'] if item['status'] == 'TRADING']
                symbols.extend(spot_symbols)
            
            # Obtener símbolos de futuros si es necesario
            if asset_type in ['futures', 'all', None]:
                futures_url = f"{self.futures_url}/exchangeInfo"
                futures_response = requests.get(futures_url)
                futures_response.raise_for_status()
                futures_data = futures_response.json()
                
                futures_symbols = [item['symbol'] for item in futures_data['symbols'] if item['status'] == 'TRADING']
                # Añadir sufijo _PERP para distinguir los futuros
                futures_symbols = [f"{s}_PERP" for s in futures_symbols]
                symbols.extend(futures_symbols)
            
            logger.info(f"Obtenidos {len(symbols)} símbolos de Binance")
            return symbols
            
        except Exception as e:
            logger.error(f"Error al obtener lista de símbolos de Binance: {str(e)}")
            return []

class DataManager:
    """
    Clase principal para gestionar la obtención de datos de diferentes fuentes
    """
    
    def __init__(self, default_provider='yahoo'):
        """
        Inicializa el gestor de datos
        
        Args:
            default_provider (str): Proveedor de datos por defecto ('yahoo', 'binance')
        """
        self.providers = {}
        self.default_provider = default_provider
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 300  # 5 minutos en segundos
        
        # Inicializar proveedores
        self.providers['yahoo'] = YahooFinanceProvider()
        self.providers['binance'] = BinanceProvider()
        
        logger.info(f"DataManager inicializado con proveedor por defecto: {default_provider}")
    
    def get_provider(self, provider_name=None):
        """
        Obtiene una instancia del proveedor de datos
        
        Args:
            provider_name (str, optional): Nombre del proveedor
            
        Returns:
            DataProvider: Instancia del proveedor de datos
        """
        provider = provider_name or self.default_provider
        if provider not in self.providers:
            logger.warning(f"Proveedor {provider} no encontrado, usando {self.default_provider}")
            provider = self.default_provider
        
        return self.providers[provider]
    
    def get_historical_data(self, symbol, interval='1d', start_time=None, end_time=None, provider=None):
        """
      
(Content truncated due to size limit. Use line ranges to read in chunks)