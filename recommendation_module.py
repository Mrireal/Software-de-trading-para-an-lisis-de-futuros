"""
Módulo de Recomendaciones para el Software de Trading en Futuros

Este módulo implementa algoritmos avanzados para generar recomendaciones
de compra y venta basadas en análisis técnico, patrones de mercado y
estrategias personalizadas.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import os

# Importar nuestros módulos
from data_module import DataManager
from analysis_module import TechnicalAnalysis
import data_utils as utils

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_recommendations.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("recommendations_module")

class TradingRecommendationEngine:
    """Clase principal para generar recomendaciones de trading"""
    
    def __init__(self, risk_profile="moderate"):
        """
        Inicializa el motor de recomendaciones
        
        Args:
            risk_profile (str): Perfil de riesgo ('conservative', 'moderate', 'aggressive')
        """
        self.risk_profile = risk_profile
        self.data_manager = DataManager()
        self.analyzer = TechnicalAnalysis()
        self.strategies = {}
        self.load_strategies()
        logger.info(f"Motor de recomendaciones inicializado con perfil de riesgo: {risk_profile}")
    
    def load_strategies(self):
        """Carga estrategias predefinidas desde archivo o inicializa por defecto"""
        strategies_file = "strategies.json"
        
        if os.path.exists(strategies_file):
            try:
                with open(strategies_file, 'r') as f:
                    self.strategies = json.load(f)
                logger.info(f"Estrategias cargadas desde {strategies_file}")
            except Exception as e:
                logger.error(f"Error al cargar estrategias: {str(e)}")
                self.initialize_default_strategies()
        else:
            logger.info("Archivo de estrategias no encontrado, inicializando estrategias por defecto")
            self.initialize_default_strategies()
    
    def initialize_default_strategies(self):
        """Inicializa estrategias predefinidas por defecto"""
        self.strategies = {
            "conservative": {
                "name": "Estrategia Conservadora",
                "description": "Enfocada en minimizar riesgos con operaciones de menor frecuencia",
                "indicators": {
                    "sma": {"periods": [50, 200], "weight": 2.0},
                    "rsi": {"overbought": 75, "oversold": 25, "weight": 1.5},
                    "bollinger": {"std_dev": 2.5, "weight": 1.5},
                    "volume": {"min_ratio": 1.5, "weight": 1.0}
                },
                "filters": {
                    "min_signal_strength": 3,
                    "confirmation_period": 3,
                    "stop_loss_pct": 5,
                    "take_profit_pct": 10
                }
            },
            "moderate": {
                "name": "Estrategia Moderada",
                "description": "Balance entre riesgo y recompensa con operaciones regulares",
                "indicators": {
                    "sma": {"periods": [20, 50], "weight": 1.5},
                    "macd": {"weight": 2.0},
                    "rsi": {"overbought": 70, "oversold": 30, "weight": 1.5},
                    "bollinger": {"std_dev": 2.0, "weight": 1.0},
                    "volume": {"min_ratio": 1.2, "weight": 1.0}
                },
                "filters": {
                    "min_signal_strength": 2,
                    "confirmation_period": 2,
                    "stop_loss_pct": 7,
                    "take_profit_pct": 15
                }
            },
            "aggressive": {
                "name": "Estrategia Agresiva",
                "description": "Enfocada en maximizar ganancias con operaciones frecuentes",
                "indicators": {
                    "ema": {"periods": [9, 21], "weight": 1.5},
                    "macd": {"weight": 1.5},
                    "rsi": {"overbought": 65, "oversold": 35, "weight": 1.0},
                    "stochastic": {"weight": 1.5},
                    "adx": {"min_value": 20, "weight": 1.0},
                    "volume": {"min_ratio": 1.0, "weight": 1.0}
                },
                "filters": {
                    "min_signal_strength": 1,
                    "confirmation_period": 1,
                    "stop_loss_pct": 10,
                    "take_profit_pct": 20
                }
            }
        }
        
        # Guardar estrategias en archivo
        try:
            with open("strategies.json", 'w') as f:
                json.dump(self.strategies, f, indent=4)
            logger.info("Estrategias por defecto guardadas en strategies.json")
        except Exception as e:
            logger.error(f"Error al guardar estrategias: {str(e)}")
    
    def set_risk_profile(self, risk_profile):
        """
        Establece el perfil de riesgo
        
        Args:
            risk_profile (str): Perfil de riesgo ('conservative', 'moderate', 'aggressive')
        """
        if risk_profile in self.strategies:
            self.risk_profile = risk_profile
            logger.info(f"Perfil de riesgo cambiado a: {risk_profile}")
        else:
            logger.warning(f"Perfil de riesgo '{risk_profile}' no válido, usando 'moderate'")
            self.risk_profile = "moderate"
    
    def get_data_with_analysis(self, symbol, interval='1d', start_time=None, provider=None):
        """
        Obtiene datos con análisis técnico aplicado
        
        Args:
            symbol (str): Símbolo del activo
            interval (str): Intervalo de tiempo
            start_time (str/datetime): Tiempo de inicio
            provider (str): Proveedor de datos
            
        Returns:
            pandas.DataFrame: DataFrame con datos e indicadores
        """
        # Si no se especifica start_time, usar 6 meses por defecto
        if start_time is None:
            start_time = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        
        # Obtener datos
        df = self.data_manager.get_historical_data(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            provider=provider
        )
        
        if df is None or df.empty:
            logger.warning(f"No se pudieron obtener datos para {symbol}")
            return None
        
        # Limpiar datos
        df = utils.clean_dataframe(df)
        
        # Aplicar análisis técnico según el perfil de riesgo
        strategy = self.strategies.get(self.risk_profile, self.strategies["moderate"])
        
        # Añadir indicadores según la estrategia
        indicators = strategy["indicators"]
        
        # SMA
        if "sma" in indicators:
            df = self.analyzer.add_moving_averages(df, windows=indicators["sma"]["periods"])
        
        # EMA
        if "ema" in indicators:
            df = self.analyzer.add_exponential_moving_averages(df, windows=indicators["ema"]["periods"])
        
        # MACD
        if "macd" in indicators:
            df = self.analyzer.add_macd(df)
        
        # RSI
        if "rsi" in indicators:
            df = self.analyzer.add_rsi(df)
        
        # Bollinger Bands
        if "bollinger" in indicators:
            std_dev = indicators["bollinger"].get("std_dev", 2.0)
            df = self.analyzer.add_bollinger_bands(df, std_dev=std_dev)
        
        # Stochastic
        if "stochastic" in indicators:
            df = self.analyzer.add_stochastic_oscillator(df)
        
        # ADX
        if "adx" in indicators:
            df = self.analyzer.add_adx(df)
        
        # Volume
        if "volume" in indicators:
            df = self.analyzer.add_volume_indicators(df)
        
        # Generar señales
        df = self.analyzer.generate_signals(df)
        
        return df
    
    def generate_recommendation(self, symbol, interval='1d', start_time=None, provider=None):
        """
        Genera una recomendación de trading para un símbolo específico
        
        Args:
            symbol (str): Símbolo del activo
            interval (str): Intervalo de tiempo
            start_time (str/datetime): Tiempo de inicio
            provider (str): Proveedor de datos
            
        Returns:
            dict: Recomendación generada
        """
        # Obtener datos con análisis
        df = self.get_data_with_analysis(symbol, interval, start_time, provider)
        
        if df is None or df.empty:
            return {
                "symbol": symbol,
                "recommendation": "NEUTRAL",
                "confidence": 0,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "reason": "No hay datos suficientes para generar una recomendación",
                "timestamp": datetime.now().isoformat()
            }
        
        # Obtener precio actual
        current_price = self.data_manager.get_current_price(symbol, provider)
        
        # Aplicar estrategia según perfil de riesgo
        recommendation = self.apply_strategy(df, current_price, symbol)
        
        return recommendation
    
    def apply_strategy(self, df, current_price, symbol):
        """
        Aplica la estrategia seleccionada para generar una recomendación
        
        Args:
            df (pandas.DataFrame): DataFrame con datos e indicadores
            current_price (float): Precio actual
            symbol (str): Símbolo del activo
            
        Returns:
            dict: Recomendación generada
        """
        # Obtener estrategia según perfil de riesgo
        strategy = self.strategies.get(self.risk_profile, self.strategies["moderate"])
        
        # Inicializar puntuación
        buy_score = 0
        sell_score = 0
        reasons = []
        
        # Obtener últimas filas para análisis
        last_row = df.iloc[-1]
        
        # Verificar señales recientes
        recent_df = df.tail(strategy["filters"]["confirmation_period"])
        recent_signals = recent_df[recent_df['signal'] != 0]
        
        # Si hay señales recientes, añadir puntuación
        if not recent_signals.empty:
            for _, signal_row in recent_signals.iterrows():
                if signal_row['signal'] == 1:  # Señal de compra
                    buy_score += 1 * signal_row['signal_strength']
                    reasons.append(f"Señal de compra detectada el {signal_row['timestamp'].strftime('%Y-%m-%d')}")
                elif signal_row['signal'] == -1:  # Señal de venta
                    sell_score += 1 * signal_row['signal_strength']
                    reasons.append(f"Señal de venta detectada el {signal_row['timestamp'].strftime('%Y-%m-%d')}")
        
        # Analizar indicadores según la estrategia
        indicators = strategy["indicators"]
        
        # SMA
        if "sma" in indicators:
            sma_weight = indicators["sma"]["weight"]
            sma_periods = indicators["sma"]["periods"]
            
            # Verificar cruces de SMA
            if len(sma_periods) >= 2:
                short_sma = f"sma_{sma_periods[0]}"
                long_sma = f"sma_{sma_periods[1]}"
                
                if short_sma in df.columns and long_sma in df.columns:
                    # Cruce alcista (corto por encima de largo)
                    if last_row[short_sma] > last_row[long_sma] and df[short_sma].iloc[-2] <= df[long_sma].iloc[-2]:
                        buy_score += sma_weight
                        reasons.append(f"Cruce alcista de SMA {sma_periods[0]} por encima de SMA {sma_periods[1]}")
                    
                    # Cruce bajista (corto por debajo de largo)
                    elif last_row[short_sma] < last_row[long_sma] and df[short_sma].iloc[-2] >= df[long_sma].iloc[-2]:
                        sell_score += sma_weight
                        reasons.append(f"Cruce bajista de SMA {sma_periods[0]} por debajo de SMA {sma_periods[1]}")
                    
                    # Tendencia alcista (corto por encima de largo)
                    elif last_row[short_sma] > last_row[long_sma]:
                        buy_score += sma_weight * 0.5
                        reasons.append(f"Tendencia alcista: SMA {sma_periods[0]} por encima de SMA {sma_periods[1]}")
                    
                    # Tendencia bajista (corto por debajo de largo)
                    elif last_row[short_sma] < last_row[long_sma]:
                        sell_score += sma_weight * 0.5
                        reasons.append(f"Tendencia bajista: SMA {sma_periods[0]} por debajo de SMA {sma_periods[1]}")
        
        # EMA
        if "ema" in indicators:
            ema_weight = indicators["ema"]["weight"]
            ema_periods = indicators["ema"]["periods"]
            
            # Verificar cruces de EMA
            if len(ema_periods) >= 2:
                short_ema = f"ema_{ema_periods[0]}"
                long_ema = f"ema_{ema_periods[1]}"
                
                if short_ema in df.columns and long_ema in df.columns:
                    # Cruce alcista (corto por encima de largo)
                    if last_row[short_ema] > last_row[long_ema] and df[short_ema].iloc[-2] <= df[long_ema].iloc[-2]:
                        buy_score += ema_weight
                        reasons.append(f"Cruce alcista de EMA {ema_periods[0]} por encima de EMA {ema_periods[1]}")
                    
                    # Cruce bajista (corto por debajo de largo)
                    elif last_row[short_ema] < last_row[long_ema] and df[short_ema].iloc[-2] >= df[long_ema].iloc[-2]:
                        sell_score += ema_weight
                        reasons.append(f"Cruce bajista de EMA {ema_periods[0]} por debajo de EMA {ema_periods[1]}")
                    
                    # Tendencia alcista (corto por encima de largo)
                    elif last_row[short_ema] > last_row[long_ema]:
                        buy_score += ema_weight * 0.5
                        reasons.append(f"Tendencia alcista: EMA {ema_periods[0]} por encima de EMA {ema_periods[1]}")
                    
                    # Tendencia bajista (corto por debajo de largo)
                    elif last_row[short_ema] < last_row[long_ema]:
                        sell_score += ema_weight * 0.5
                        reasons.append(f"Tendencia bajista: EMA {ema_periods[0]} por debajo de EMA {ema_periods[1]}")
        
        # RSI
        if "rsi" in indicators and 'rsi' in df.columns:
            rsi_weight = indicators["rsi"]["weight"]
            oversold = indicators["rsi"]["oversold"]
            overbought = indicators["rsi"]["overbought"]
            
            # Sobreventa (señal de compra)
            if last_row['rsi'] < oversold:
                buy_score += rsi_weight
                reasons.append(f"RSI en zona de sobreventa ({last_row['rsi']:.2f} < {oversold})")
            
            # Sobrecompra (señal de venta)
            elif last_row['rsi'] > overbought:
                sell_score += rsi_weight
                reasons.append(f"RSI en zona de sobrecompra ({last_row['rsi']:.2f} > {overbought})")
            
            # Saliendo de sobreventa (señal de compra)
            elif last_row['rsi'] > oversold and df['rsi'].iloc[-2] <= oversold:
                buy_score += rsi_weight * 0.8
                reasons.append(f"RSI saliendo de zona de sobreventa ({last_row['rsi']:.2f})")
            
            # Entrando en sobrecompra (señal de venta)
            elif last_row['rsi'] < overbought and df['rsi'].iloc[-2] >= overbought:
                sell_score += rsi_weight * 0.8
                reasons.append(f"RSI saliendo de zona 
(Content truncated due to size limit. Use line ranges to read in chunks)