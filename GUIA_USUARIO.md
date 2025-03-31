# Guía de Usuario - Software de Trading en Futuros

## Introducción

Bienvenido al Software de Trading en Futuros, una herramienta completa diseñada para ayudarle a analizar mercados financieros y recibir recomendaciones de compra/venta basadas en análisis técnico avanzado. Esta guía le proporcionará toda la información necesaria para utilizar el software de manera efectiva.

## Primeros Pasos

### Instalación

1. Asegúrese de tener Python 3.8 o superior instalado en su sistema.
2. Instale las dependencias necesarias:
   ```
   pip install pandas numpy matplotlib streamlit plotly requests
   ```
3. Ejecute la aplicación:
   ```
   python app.py
   ```
4. Acceda a la interfaz web en su navegador: http://localhost:8501

## Interfaz de Usuario

La interfaz está organizada en varias secciones para facilitar la navegación y el análisis:

### Barra Lateral

- **Selección de Activo**: Elija el activo que desea analizar (criptomonedas, acciones o futuros)
- **Intervalo de Tiempo**: Seleccione la granularidad de los datos (1 minuto a 1 mes)
- **Período de Análisis**: Elija el rango de fechas para el análisis
- **Perfil de Riesgo**: Seleccione entre conservador, moderado o agresivo
- **Indicadores Técnicos**: Active/desactive los indicadores que desea visualizar

### Panel Principal

El panel principal está organizado en pestañas:

1. **Análisis de Mercado**: Muestra el gráfico de precios con indicadores técnicos
2. **Señales de Trading**: Presenta las señales de compra/venta identificadas
3. **Recomendaciones**: Muestra recomendaciones actuales con niveles de entrada, stop loss y take profit
4. **Backtesting**: Permite evaluar el rendimiento histórico de diferentes estrategias
5. **Configuración**: Opciones avanzadas para personalizar el software

## Uso Básico

### Analizar un Activo

1. En la barra lateral, seleccione el activo que desea analizar (por ejemplo, "BTC-USD" para Bitcoin)
2. Elija el intervalo de tiempo adecuado (por ejemplo, "1d" para datos diarios)
3. Seleccione el período de análisis (por ejemplo, "3mo" para tres meses)
4. El gráfico se actualizará automáticamente mostrando el precio y los indicadores seleccionados

### Obtener Recomendaciones

1. Seleccione su perfil de riesgo en la barra lateral
2. Vaya a la pestaña "Recomendaciones"
3. El sistema mostrará:
   - Recomendación actual (COMPRAR, VENDER o NEUTRAL)
   - Nivel de confianza de la recomendación
   - Precio de entrada sugerido
   - Niveles de stop loss y take profit
   - Razones que justifican la recomendación

### Realizar Backtesting

1. Vaya a la pestaña "Backtesting"
2. Seleccione la estrategia que desea evaluar
3. Ajuste los parámetros de la estrategia si es necesario
4. Haga clic en "Ejecutar Backtesting"
5. Analice los resultados:
   - Rendimiento total de la estrategia
   - Comparación con estrategia de comprar y mantener
   - Número de operaciones realizadas
   - Ratio de operaciones ganadoras
   - Máximo drawdown

## Funciones Avanzadas

### Personalizar Indicadores Técnicos

1. Vaya a la pestaña "Configuración"
2. En la sección "Indicadores Técnicos", puede:
   - Modificar los períodos de los indicadores
   - Ajustar los umbrales para las señales
   - Cambiar los colores de visualización

### Crear Estrategias Personalizadas

1. Vaya a la pestaña "Configuración"
2. En la sección "Estrategias", puede:
   - Crear nuevas estrategias combinando diferentes indicadores
   - Asignar pesos a cada indicador
   - Definir reglas de entrada y salida
   - Establecer parámetros de gestión de riesgo

### Exportar Datos y Resultados

1. En cualquier pestaña, encontrará botones para:
   - Exportar datos a CSV
   - Guardar gráficos como imágenes
   - Generar informes en PDF

## Interpretación de Resultados

### Señales de Trading

- **Señal de Compra (verde)**: Indica una oportunidad potencial de compra
- **Señal de Venta (rojo)**: Indica una oportunidad potencial de venta
- **Fuerza de la Señal**: Representada por la intensidad del color y el tamaño del marcador

### Recomendaciones

- **Nivel de Confianza**: Porcentaje que indica la fuerza de la recomendación
  - 0-33%: Baja confianza
  - 34-66%: Confianza moderada
  - 67-100%: Alta confianza
- **Razones**: Explicación de los factores que han generado la recomendación

### Resultados de Backtesting

- **Rendimiento Total**: Ganancia o pérdida porcentual durante el período analizado
- **Ratio de Ganancia**: Proporción de operaciones ganadoras respecto al total
- **Drawdown Máximo**: Mayor caída desde un pico hasta un valle, indicador de riesgo

## Consejos y Mejores Prácticas

1. **Diversifique su análisis**: No se base únicamente en un indicador o estrategia
2. **Valide las señales**: Confirme las señales con múltiples indicadores
3. **Gestione el riesgo**: Utilice siempre stop loss para limitar pérdidas potenciales
4. **Pruebe antes de operar**: Realice backtesting exhaustivo antes de aplicar estrategias con dinero real
5. **Actualice regularmente**: Los mercados cambian, ajuste sus estrategias periódicamente

## Solución de Problemas

### Problemas Comunes

1. **Datos no disponibles**: 
   - Verifique su conexión a internet
   - Asegúrese de que el símbolo es correcto
   - Algunos datos pueden tener retraso debido a limitaciones de las APIs gratuitas

2. **Rendimiento lento**:
   - Reduzca el período de análisis
   - Desactive indicadores que no esté utilizando
   - Cierre otras aplicaciones que consuman muchos recursos

3. **Recomendaciones inconsistentes**:
   - Las recomendaciones se basan en análisis técnico, que tiene limitaciones
   - Mercados muy volátiles pueden generar señales contradictorias
   - Utilice múltiples períodos de tiempo para confirmar tendencias

## Advertencias

- Este software proporciona análisis técnico y recomendaciones basadas en datos históricos
- El trading de futuros y criptomonedas implica riesgos significativos
- Las recomendaciones no garantizan resultados futuros
- Utilice este software como una herramienta de apoyo, no como único criterio para sus decisiones de inversión
- Consulte con un asesor financiero antes de realizar operaciones con dinero real

## Soporte y Contacto

Si encuentra problemas o tiene sugerencias para mejorar el software, por favor:
- Consulte la documentación en README.md
- Revise los archivos de registro en caso de errores
- Contacte al desarrollador a través del repositorio de GitHub

---

Gracias por utilizar nuestro Software de Trading en Futuros. ¡Le deseamos éxito en sus operaciones!
