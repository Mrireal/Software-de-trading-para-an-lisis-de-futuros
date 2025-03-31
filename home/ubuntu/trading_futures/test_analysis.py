"""
Script para probar el módulo de análisis técnico con datos reales.
Este script demuestra cómo utilizar el módulo de análisis para generar
indicadores técnicos y señales de trading para diferentes activos.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Importar nuestros módulos
from data_module import DataManager
from analysis_module import TechnicalAnalysis
import data_utils as utils

def test_technical_analysis(symbol, provider='yahoo', interval='1d', days=90):
    """
    Prueba el análisis técnico para un símbolo específico
    
    Args:
        symbol (str): Símbolo a analizar
        provider (str): Proveedor de datos ('yahoo', 'binance')
        interval (str): Intervalo de tiempo
        days (int): Número de días de datos históricos
    """
    print(f"\n{'='*50}")
    print(f"Análisis técnico para: {symbol} usando {provider}")
    print(f"{'='*50}")
    
    # Crear directorio para gráficos
    os.makedirs('analysis_charts', exist_ok=True)
    
    # Obtener datos
    data_manager = DataManager(default_provider=provider)
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    print(f"Obteniendo datos históricos desde {start_date}...")
    df = data_manager.get_historical_data(symbol, interval=interval, start_time=start_date)
    
    if df is None or df.empty:
        print(f"No se pudieron obtener datos para {symbol}")
        return
    
    # Limpiar datos
    df = utils.clean_dataframe(df)
    
    # Crear instancia del analizador técnico
    analyzer = TechnicalAnalysis()
    
    # Añadir indicadores básicos
    print("Calculando indicadores técnicos...")
    df = analyzer.add_moving_averages(df)
    df = analyzer.add_macd(df)
    df = analyzer.add_rsi(df)
    df = analyzer.add_bollinger_bands(df)
    
    # Generar señales
    print("Generando señales de trading...")
    df = analyzer.generate_signals(df)
    
    # Mostrar últimas señales
    recent_signals = df[df['signal'] != 0].tail(5)
    if not recent_signals.empty:
        print("\nÚltimas señales generadas:")
        for idx, row in recent_signals.iterrows():
            signal_type = "COMPRA" if row['signal'] == 1 else "VENTA"
            signal_date = row['timestamp'].strftime('%Y-%m-%d')
            signal_price = row['close']
            signal_strength = row['signal_strength']
            print(f"  {signal_date}: Señal de {signal_type} a {signal_price:.2f} (Fuerza: {signal_strength})")
    else:
        print("\nNo se generaron señales en el período analizado")
    
    # Visualizar resultados
    print("\nGenerando gráfico de análisis técnico...")
    chart_path = f'analysis_charts/{symbol.replace("/", "_")}_{provider}_{datetime.now().strftime("%Y%m%d")}.png'
    analyzer.plot_analysis(df, symbol, save_path=chart_path)
    
    print(f"Análisis técnico completado y guardado en {chart_path}")
    return df

def compare_strategies(symbol, provider='yahoo', interval='1d', days=180):
    """
    Compara diferentes estrategias de trading basadas en indicadores técnicos
    
    Args:
        symbol (str): Símbolo a analizar
        provider (str): Proveedor de datos
        interval (str): Intervalo de tiempo
        days (int): Número de días de datos históricos
    """
    print(f"\n{'='*50}")
    print(f"Comparación de estrategias para: {symbol}")
    print(f"{'='*50}")
    
    # Crear directorio para gráficos
    os.makedirs('analysis_charts', exist_ok=True)
    
    # Obtener datos
    data_manager = DataManager(default_provider=provider)
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    print(f"Obteniendo datos históricos desde {start_date}...")
    df = data_manager.get_historical_data(symbol, interval=interval, start_time=start_date)
    
    if df is None or df.empty:
        print(f"No se pudieron obtener datos para {symbol}")
        return
    
    # Limpiar datos
    df = utils.clean_dataframe(df)
    
    # Crear instancia del analizador técnico
    analyzer = TechnicalAnalysis()
    
    # Añadir todos los indicadores
    print("Calculando indicadores técnicos...")
    df = analyzer.add_all_indicators(df)
    
    # Definir diferentes estrategias
    strategies = {
        'moving_avg_cross': create_ma_crossover_strategy(df),
        'rsi_strategy': create_rsi_strategy(df),
        'macd_strategy': create_macd_strategy(df),
        'bollinger_strategy': create_bollinger_strategy(df)
    }
    
    # Evaluar rendimiento de cada estrategia
    print("\nEvaluando rendimiento de estrategias:")
    for name, strategy_df in strategies.items():
        performance = calculate_strategy_performance(strategy_df)
        print(f"  {name.upper()}: Rendimiento: {performance['return']:.2f}%, Operaciones: {performance['trades']}, Ratio Ganancia/Pérdida: {performance['win_ratio']:.2f}")
    
    # Visualizar comparación
    print("\nGenerando gráfico comparativo...")
    chart_path = f'analysis_charts/{symbol.replace("/", "_")}_strategies_comparison_{datetime.now().strftime("%Y%m%d")}.png'
    plot_strategies_comparison(strategies, symbol, chart_path)
    
    print(f"Comparación de estrategias completada y guardada en {chart_path}")
    return strategies

def create_ma_crossover_strategy(df):
    """Crea una estrategia basada en cruce de medias móviles"""
    df_strategy = df.copy()
    df_strategy['strategy_signal'] = 0
    
    # Señal de compra: SMA corta cruza por encima de SMA larga
    if all(col in df_strategy.columns for col in ['sma_20', 'sma_50']):
        df_strategy.loc[(df_strategy['sma_20'] > df_strategy['sma_50']) & 
                      (df_strategy['sma_20'].shift() <= df_strategy['sma_50'].shift()), 'strategy_signal'] = 1
        
        # Señal de venta: SMA corta cruza por debajo de SMA larga
        df_strategy.loc[(df_strategy['sma_20'] < df_strategy['sma_50']) & 
                      (df_strategy['sma_20'].shift() >= df_strategy['sma_50'].shift()), 'strategy_signal'] = -1
    
    # Calcular rendimiento
    calculate_strategy_returns(df_strategy)
    return df_strategy

def create_rsi_strategy(df):
    """Crea una estrategia basada en RSI"""
    df_strategy = df.copy()
    df_strategy['strategy_signal'] = 0
    
    if 'rsi' in df_strategy.columns:
        # Señal de compra: RSI por debajo de 30
        df_strategy.loc[df_strategy['rsi'] < 30, 'strategy_signal'] = 1
        
        # Señal de venta: RSI por encima de 70
        df_strategy.loc[df_strategy['rsi'] > 70, 'strategy_signal'] = -1
    
    # Calcular rendimiento
    calculate_strategy_returns(df_strategy)
    return df_strategy

def create_macd_strategy(df):
    """Crea una estrategia basada en MACD"""
    df_strategy = df.copy()
    df_strategy['strategy_signal'] = 0
    
    if all(col in df_strategy.columns for col in ['macd', 'macd_signal']):
        # Señal de compra: MACD cruza por encima de la línea de señal
        df_strategy.loc[(df_strategy['macd'] > df_strategy['macd_signal']) & 
                      (df_strategy['macd'].shift() <= df_strategy['macd_signal'].shift()), 'strategy_signal'] = 1
        
        # Señal de venta: MACD cruza por debajo de la línea de señal
        df_strategy.loc[(df_strategy['macd'] < df_strategy['macd_signal']) & 
                      (df_strategy['macd'].shift() >= df_strategy['macd_signal'].shift()), 'strategy_signal'] = -1
    
    # Calcular rendimiento
    calculate_strategy_returns(df_strategy)
    return df_strategy

def create_bollinger_strategy(df):
    """Crea una estrategia basada en Bandas de Bollinger"""
    df_strategy = df.copy()
    df_strategy['strategy_signal'] = 0
    
    if all(col in df_strategy.columns for col in ['close', 'bb_lower', 'bb_upper']):
        # Señal de compra: Precio toca la banda inferior
        df_strategy.loc[df_strategy['close'] <= df_strategy['bb_lower'], 'strategy_signal'] = 1
        
        # Señal de venta: Precio toca la banda superior
        df_strategy.loc[df_strategy['close'] >= df_strategy['bb_upper'], 'strategy_signal'] = -1
    
    # Calcular rendimiento
    calculate_strategy_returns(df_strategy)
    return df_strategy

def calculate_strategy_returns(df):
    """Calcula los retornos de una estrategia"""
    df['position'] = df['strategy_signal'].shift(1).fillna(0)
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['position'] * df['returns']
    df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
    df['cumulative_strategy_returns'] = (1 + df['strategy_returns']).cumprod() - 1
    return df

def calculate_strategy_performance(df):
    """Calcula métricas de rendimiento para una estrategia"""
    # Número de operaciones
    trades = len(df[df['strategy_signal'] != 0])
    
    # Rendimiento total
    if 'cumulative_strategy_returns' in df.columns and len(df) > 0:
        total_return = df['cumulative_strategy_returns'].iloc[-1] * 100
    else:
        total_return = 0
    
    # Ratio de operaciones ganadoras
    winning_trades = len(df[df['strategy_returns'] > 0])
    win_ratio = winning_trades / trades if trades > 0 else 0
    
    return {
        'trades': trades,
        'return': total_return,
        'win_ratio': win_ratio
    }

def plot_strategies_comparison(strategies, symbol, save_path=None):
    """Genera un gráfico comparativo de diferentes estrategias"""
    plt.figure(figsize=(14, 8))
    
    # Gráfico de rendimientos acumulados
    for name, df in strategies.items():
        if 'cumulative_strategy_returns' in df.columns:
            plt.plot(df['timestamp'], df['cumulative_strategy_returns'] * 100, label=name)
    
    # Añadir rendimiento de buy & hold
    if 'cumulative_returns' in list(strategies.values())[0].columns:
        plt.plot(list(strategies.values())[0]['timestamp'], 
                list(strategies.values())[0]['cumulative_returns'] * 100, 
                label='Buy & Hold', linestyle='--')
    
    plt.title(f'Comparación de Estrategias para {symbol}')
    plt.xlabel('Fecha')
    plt.ylabel('Rendimiento (%)')
    plt.grid(True)
    plt.legend()
    
    # Guardar gráfico si se especifica ruta
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def main():
    """Función principal para probar el módulo de análisis técnico"""
    # Probar análisis técnico con diferentes activos
    test_technical_analysis('BTC-USD', provider='yahoo', days=90)
    test_technical_analysis('ETH-USD', provider='yahoo', days=90)
    test_technical_analysis('AAPL', provider='yahoo', days=90)
    
    # Comparar estrategias
    compare_strategies('BTC-USD', provider='yahoo', days=180)
    
    print("\nPruebas de análisis técnico completadas con éxito!")

if __name__ == "__main__":
    main()
