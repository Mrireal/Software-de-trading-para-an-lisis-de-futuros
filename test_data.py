"""
Módulo principal para probar la funcionalidad de obtención de datos del mercado.
Este script demuestra cómo utilizar el módulo de datos para obtener información
financiera de diferentes fuentes y procesarla para su análisis.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Importar nuestros módulos
from data_module import DataManager
import data_utils as utils

def test_data_provider(provider_name, symbol, interval='1d', days=30):
    """
    Prueba un proveedor de datos específico
    
    Args:
        provider_name (str): Nombre del proveedor ('yahoo', 'binance')
        symbol (str): Símbolo a consultar
        interval (str): Intervalo de tiempo
        days (int): Número de días de datos históricos
    """
    print(f"\n{'='*50}")
    print(f"Probando proveedor: {provider_name.upper()} con símbolo: {symbol}")
    print(f"{'='*50}")
    
    # Crear gestor de datos con el proveedor especificado
    data_manager = DataManager(default_provider=provider_name)
    
    # Calcular fecha de inicio
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # Obtener datos históricos
    print(f"Obteniendo datos históricos desde {start_date}...")
    df = data_manager.get_historical_data(symbol, interval=interval, start_time=start_date)
    
    if df is None or df.empty:
        print(f"No se pudieron obtener datos para {symbol}")
        return
    
    # Validar y limpiar datos
    if not utils.validate_dataframe(df):
        print("Los datos obtenidos no tienen el formato esperado")
        return
    
    df = utils.clean_dataframe(df)
    
    # Mostrar información básica
    print(f"Datos obtenidos: {len(df)} registros")
    print("\nPrimeros registros:")
    print(df.head())
    
    print("\nÚltimos registros:")
    print(df.tail())
    
    print("\nEstadísticas descriptivas:")
    print(df.describe())
    
    # Obtener precio actual
    current_price = data_manager.get_current_price(symbol)
    print(f"\nPrecio actual de {symbol}: {current_price}")
    
    # Calcular indicadores adicionales
    df = utils.calculate_returns(df)
    df = utils.calculate_volatility(df)
    
    # Visualizar datos
    plot_data(df, symbol, provider_name)
    
    return df

def plot_data(df, symbol, provider_name):
    """
    Visualiza los datos financieros
    
    Args:
        df (pandas.DataFrame): DataFrame con datos financieros
        symbol (str): Símbolo del activo
        provider_name (str): Nombre del proveedor
    """
    # Crear directorio para gráficos si no existe
    os.makedirs('charts', exist_ok=True)
    
    # Configurar el gráfico
    plt.figure(figsize=(12, 8))
    
    # Gráfico de precios
    plt.subplot(2, 1, 1)
    plt.plot(df['timestamp'], df['close'], label='Precio de cierre')
    plt.title(f'Precio histórico de {symbol} - Datos de {provider_name.upper()}')
    plt.xlabel('Fecha')
    plt.ylabel('Precio')
    plt.grid(True)
    plt.legend()
    
    # Gráfico de volumen
    plt.subplot(2, 1, 2)
    plt.bar(df['timestamp'], df['volume'], alpha=0.7, label='Volumen')
    plt.title(f'Volumen de {symbol}')
    plt.xlabel('Fecha')
    plt.ylabel('Volumen')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Guardar gráfico
    filename = f'charts/{symbol}_{provider_name}_{datetime.now().strftime("%Y%m%d")}.png'
    plt.savefig(filename)
    print(f"\nGráfico guardado en: {filename}")
    plt.close()

def compare_providers(symbol, interval='1d', days=30):
    """
    Compara datos de diferentes proveedores para el mismo símbolo
    
    Args:
        symbol (str): Símbolo a consultar
        interval (str): Intervalo de tiempo
        days (int): Número de días de datos históricos
    """
    print(f"\n{'='*50}")
    print(f"Comparando proveedores para símbolo: {symbol}")
    print(f"{'='*50}")
    
    # Obtener datos de diferentes proveedores
    yahoo_data = test_data_provider('yahoo', symbol, interval, days)
    
    # Para Binance, ajustar el símbolo si es necesario
    binance_symbol = symbol
    if '-USD' in symbol:
        binance_symbol = symbol.replace('-USD', 'USDT')
    
    binance_data = test_data_provider('binance', binance_symbol, interval, days)
    
    # Si ambos proveedores devolvieron datos, comparar
    if yahoo_data is not None and not yahoo_data.empty and binance_data is not None and not binance_data.empty:
        # Crear directorio para gráficos si no existe
        os.makedirs('charts', exist_ok=True)
        
        # Configurar el gráfico
        plt.figure(figsize=(12, 6))
        
        # Normalizar datos para comparación justa
        yahoo_norm = yahoo_data['close'] / yahoo_data['close'].iloc[0]
        binance_norm = binance_data['close'] / binance_data['close'].iloc[0]
        
        # Gráfico de comparación
        plt.plot(yahoo_data['timestamp'], yahoo_norm, label='Yahoo Finance')
        plt.plot(binance_data['timestamp'], binance_norm, label='Binance')
        plt.title(f'Comparación de proveedores para {symbol} (normalizado)')
        plt.xlabel('Fecha')
        plt.ylabel('Precio normalizado')
        plt.grid(True)
        plt.legend()
        
        # Guardar gráfico
        filename = f'charts/{symbol}_comparison_{datetime.now().strftime("%Y%m%d")}.png'
        plt.savefig(filename)
        print(f"\nGráfico de comparación guardado en: {filename}")
        plt.close()

def main():
    """Función principal para probar el módulo de datos"""
    # Crear directorio para gráficos
    os.makedirs('charts', exist_ok=True)
    
    # Probar con Bitcoin
    test_data_provider('yahoo', 'BTC-USD', interval='1d', days=90)
    
    # Probar con Ethereum
    test_data_provider('yahoo', 'ETH-USD', interval='1d', days=90)
    
    # Probar con un futuro
    test_data_provider('yahoo', 'ES=F', interval='1d', days=90)
    
    # Comparar proveedores para Bitcoin
    compare_providers('BTC-USD', interval='1d', days=30)
    
    print("\nPruebas completadas con éxito!")

if __name__ == "__main__":
    main()
