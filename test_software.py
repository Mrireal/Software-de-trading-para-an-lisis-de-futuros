"""
Script para probar el software de trading en futuros con datos reales.
Este script ejecuta pruebas exhaustivas para validar el funcionamiento
de todos los componentes del software y la precisión de las recomendaciones.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import json

# Importar nuestros módulos
from data_module import DataManager
from analysis_module import TechnicalAnalysis
from recommendation_module import TradingRecommendationEngine
import data_utils as utils

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("testing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("testing_module")

def test_data_module():
    """Prueba el módulo de obtención de datos"""
    logger.info("Probando módulo de obtención de datos...")
    
    # Crear directorio para resultados
    os.makedirs("test_results", exist_ok=True)
    
    # Crear gestor de datos
    data_manager = DataManager()
    
    # Lista de símbolos para probar
    symbols = ["BTC-USD", "ETH-USD", "AAPL", "MSFT", "ES=F"]
    
    results = []
    
    for symbol in symbols:
        try:
            # Obtener datos históricos
            start_time = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            df = data_manager.get_historical_data(symbol, interval='1d', start_time=start_time)
            
            if df is None or df.empty:
                logger.warning(f"No se pudieron obtener datos para {symbol}")
                results.append({
                    "symbol": symbol,
                    "status": "ERROR",
                    "records": 0,
                    "start_date": None,
                    "end_date": None,
                    "error": "No se pudieron obtener datos"
                })
                continue
            
            # Limpiar datos
            df = utils.clean_dataframe(df)
            
            # Obtener precio actual
            current_price = data_manager.get_current_price(symbol)
            
            # Guardar resultados
            results.append({
                "symbol": symbol,
                "status": "OK",
                "records": len(df),
                "start_date": df['timestamp'].min().strftime('%Y-%m-%d') if not df.empty else None,
                "end_date": df['timestamp'].max().strftime('%Y-%m-%d') if not df.empty else None,
                "current_price": current_price
            })
            
            logger.info(f"Datos obtenidos para {symbol}: {len(df)} registros")
            
            # Guardar datos para uso posterior
            df.to_csv(f"test_results/{symbol.replace('/', '_')}_data.csv", index=False)
            
        except Exception as e:
            logger.error(f"Error al obtener datos para {symbol}: {str(e)}")
            results.append({
                "symbol": symbol,
                "status": "ERROR",
                "records": 0,
                "start_date": None,
                "end_date": None,
                "error": str(e)
            })
    
    # Guardar resultados
    with open("test_results/data_module_results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    # Crear informe
    with open("test_results/data_module_report.md", 'w') as f:
        f.write("# Informe de Pruebas: Módulo de Datos\n\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Resultados\n\n")
        f.write("| Símbolo | Estado | Registros | Fecha Inicio | Fecha Fin | Precio Actual |\n")
        f.write("|---------|--------|-----------|--------------|-----------|---------------|\n")
        
        for result in results:
            status = "✅" if result["status"] == "OK" else "❌"
            f.write(f"| {result['symbol']} | {status} | {result.get('records', 0)} | {result.get('start_date', 'N/A')} | {result.get('end_date', 'N/A')} | {result.get('current_price', 'N/A')} |\n")
        
        f.write("\n## Resumen\n\n")
        success_count = sum(1 for r in results if r["status"] == "OK")
        f.write(f"- Total de símbolos probados: {len(results)}\n")
        f.write(f"- Símbolos con datos correctos: {success_count}\n")
        f.write(f"- Símbolos con errores: {len(results) - success_count}\n")
    
    logger.info(f"Prueba del módulo de datos completada: {success_count}/{len(results)} símbolos correctos")
    
    return results

def test_analysis_module():
    """Prueba el módulo de análisis técnico"""
    logger.info("Probando módulo de análisis técnico...")
    
    # Crear directorio para resultados
    os.makedirs("test_results", exist_ok=True)
    
    # Crear analizador técnico
    analyzer = TechnicalAnalysis()
    
    # Lista de símbolos para probar
    symbols = ["BTC-USD", "ETH-USD", "AAPL", "MSFT"]
    
    results = []
    
    for symbol in symbols:
        try:
            # Cargar datos guardados o obtener nuevos si no existen
            data_file = f"test_results/{symbol.replace('/', '_')}_data.csv"
            
            if os.path.exists(data_file):
                df = pd.read_csv(data_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                # Obtener datos nuevos
                data_manager = DataManager()
                start_time = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
                df = data_manager.get_historical_data(symbol, interval='1d', start_time=start_time)
                
                if df is None or df.empty:
                    logger.warning(f"No se pudieron obtener datos para {symbol}")
                    results.append({
                        "symbol": symbol,
                        "status": "ERROR",
                        "error": "No se pudieron obtener datos"
                    })
                    continue
                
                # Limpiar datos
                df = utils.clean_dataframe(df)
                
                # Guardar datos
                df.to_csv(data_file, index=False)
            
            # Aplicar análisis técnico
            logger.info(f"Aplicando análisis técnico a {symbol}...")
            
            # Añadir indicadores
            df = analyzer.add_moving_averages(df)
            df = analyzer.add_macd(df)
            df = analyzer.add_rsi(df)
            df = analyzer.add_bollinger_bands(df)
            
            # Generar señales
            df = analyzer.generate_signals(df)
            
            # Contar señales generadas
            buy_signals = len(df[df['signal'] == 1])
            sell_signals = len(df[df['signal'] == -1])
            
            # Guardar resultados
            results.append({
                "symbol": symbol,
                "status": "OK",
                "records": len(df),
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "indicators": {
                    "sma": "sma_20" in df.columns and "sma_50" in df.columns,
                    "macd": "macd" in df.columns and "macd_signal" in df.columns,
                    "rsi": "rsi" in df.columns,
                    "bollinger": "bb_upper" in df.columns and "bb_lower" in df.columns
                }
            })
            
            logger.info(f"Análisis completado para {symbol}: {buy_signals} señales de compra, {sell_signals} señales de venta")
            
            # Guardar datos analizados
            df.to_csv(f"test_results/{symbol.replace('/', '_')}_analyzed.csv", index=False)
            
            # Generar gráfico
            analyzer.plot_analysis(df, symbol, save_path=f"test_results/{symbol.replace('/', '_')}_analysis.png")
            
        except Exception as e:
            logger.error(f"Error al analizar {symbol}: {str(e)}")
            results.append({
                "symbol": symbol,
                "status": "ERROR",
                "error": str(e)
            })
    
    # Guardar resultados
    with open("test_results/analysis_module_results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    # Crear informe
    with open("test_results/analysis_module_report.md", 'w') as f:
        f.write("# Informe de Pruebas: Módulo de Análisis Técnico\n\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Resultados\n\n")
        f.write("| Símbolo | Estado | Registros | Señales Compra | Señales Venta | SMA | MACD | RSI | Bollinger |\n")
        f.write("|---------|--------|-----------|----------------|---------------|-----|------|-----|----------|\n")
        
        for result in results:
            if result["status"] == "OK":
                status = "✅"
                sma = "✅" if result["indicators"]["sma"] else "❌"
                macd = "✅" if result["indicators"]["macd"] else "❌"
                rsi = "✅" if result["indicators"]["rsi"] else "❌"
                bollinger = "✅" if result["indicators"]["bollinger"] else "❌"
                f.write(f"| {result['symbol']} | {status} | {result['records']} | {result['buy_signals']} | {result['sell_signals']} | {sma} | {macd} | {rsi} | {bollinger} |\n")
            else:
                status = "❌"
                f.write(f"| {result['symbol']} | {status} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |\n")
        
        f.write("\n## Resumen\n\n")
        success_count = sum(1 for r in results if r["status"] == "OK")
        f.write(f"- Total de símbolos analizados: {len(results)}\n")
        f.write(f"- Símbolos analizados correctamente: {success_count}\n")
        f.write(f"- Símbolos con errores: {len(results) - success_count}\n")
        
        if success_count > 0:
            total_buy = sum(r.get("buy_signals", 0) for r in results if r["status"] == "OK")
            total_sell = sum(r.get("sell_signals", 0) for r in results if r["status"] == "OK")
            f.write(f"- Total de señales de compra generadas: {total_buy}\n")
            f.write(f"- Total de señales de venta generadas: {total_sell}\n")
    
    logger.info(f"Prueba del módulo de análisis completada: {success_count}/{len(results)} símbolos analizados correctamente")
    
    return results

def test_recommendation_module():
    """Prueba el módulo de recomendaciones"""
    logger.info("Probando módulo de recomendaciones...")
    
    # Crear directorio para resultados
    os.makedirs("test_results", exist_ok=True)
    
    # Crear motor de recomendaciones con diferentes perfiles de riesgo
    profiles = ["conservative", "moderate", "aggressive"]
    
    # Lista de símbolos para probar
    symbols = ["BTC-USD", "ETH-USD", "AAPL", "MSFT"]
    
    all_results = {}
    
    for profile in profiles:
        logger.info(f"Probando perfil de riesgo: {profile}")
        
        engine = TradingRecommendationEngine(risk_profile=profile)
        profile_results = []
        
        for symbol in symbols:
            try:
                # Generar recomendación
                recommendation = engine.generate_recommendation(symbol)
                
                # Guardar recomendación
                engine.save_recommendation_history(recommendation)
                
                # Guardar resultados
                profile_results.append({
                    "symbol": symbol,
                    "status": "OK",
                    "recommendation": recommendation["recommendation"],
                    "confidence": recommendation["confidence"],
                    "entry_price": recommendation["entry_price"],
                    "stop_loss": recommendation["stop_loss"],
                    "take_profit": recommendation["take_profit"],
                    "reason": recommendation["reason"]
                })
                
                logger.info(f"Recomendación para {symbol} ({profile}): {recommendation['recommendation']} (Confianza: {recommendation['confidence']}%)")
                
            except Exception as e:
                logger.error(f"Error al generar recomendación para {symbol} ({profile}): {str(e)}")
                profile_results.append({
                    "symbol": symbol,
                    "status": "ERROR",
                    "error": str(e)
                })
        
        all_results[profile] = profile_results
    
    # Guardar resultados
    with open("test_results/recommendation_module_results.json", 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # Crear informe
    with open("test_results/recommendation_module_report.md", 'w') as f:
        f.write("# Informe de Pruebas: Módulo de Recomendaciones\n\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for profile in profiles:
            f.write(f"## Perfil de Riesgo: {profile.capitalize()}\n\n")
            f.write("| Símbolo | Estado | Recomendación | Confianza | Precio Entrada | Stop Loss | Take Profit |\n")
            f.write("|---------|--------|--------------|-----------|----------------|-----------|-------------|\n")
            
            for result in all_results[profile]:
                if result["status"] == "OK":
                    status = "✅"
                    recommendation = result["recommendation"]
                    confidence = f"{result['confidence']}%"
                    entry_price = f"${result['entry_price']:.2f}" if result['entry_price'] else "N/A"
                    stop_loss = f"${result['stop_loss']:.2f}" if result['stop_loss'] else "N/A"
                    take_profit = f"${result['take_profit']:.2f}" if result['take_profit'] else "N/A"
                    f.write(f"| {result['symbol']} | {status} | {recommendation} | {confidence} | {entry_price} | {stop_loss} | {take_profit} |\n")
                else:
                    status = "❌"
                    f.write(f"| {result['symbol']} | {status} | N/A | N/A | N/A | N/A | N/A |\n")
            
            f.write("\n")
        
        f.write("## Resumen\n\n")
        
        for profile in profiles:
            success_count = sum(1 for r in all_results[profile] if r["status"] == "OK")
            buy_count = sum(1 for r in all_results[profile] if r["status"] == "OK" and r["recommendation"] == "COMPRAR")
            sell_count = sum(1 for r in all_results[profile] if r["status"] == "OK" and r["recommendation"] == "VENDER")
            neutral_count = sum(1 for r in all_results[profile] if r["status"] == "OK" and r["recommendation"] == "NEUTRAL")
            
            f.write(f"### Perfil {profile.capitalize()}\n")
            f.write(f"- Total de símbolos: {len(all_results[profile])}\n")
            f.write(f"- Recomendaciones generadas correctamente: {success_count}\n")
            f.write(f"- Recomendaciones de compra: {buy_count}\n")
            f.write(f"- Recomendaciones de venta: {sell_count}\n")
            f.write(f"- Recomendaciones neutrales: {neutral_count}\n\n")
    
    logger.info("Prueba del módulo de recomendaciones completada")
    
    return all_results

def test_backtest_strategy():
    """Realiza un backtest de las estrategias de trading"""
    logger.info("Realizando backtest de estrategias...")
    
    # Crear directorio para resultados
    os.makedirs("test_results", exist_ok=True)
    
    # Crear gestor de datos
    data_manager = DataManager()
    
    # Crear analizador técnico
    analyzer = TechnicalAnalysis()
    
    # Lista de símbolos para probar
    symbols = ["BTC-USD", "ETH-USD", "AAPL", "MSFT"]
    
    # Período de backtest
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Estrategias a probar
    strategies = {
        "sma_crossover": {
            "name": "Cruce de Medias Móviles",
            "description": "Compra cuando SMA corta cruza por e
(Content truncated due to size limit. Use line ranges to read in chunks)