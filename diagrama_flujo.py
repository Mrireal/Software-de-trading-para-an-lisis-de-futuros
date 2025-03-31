import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Configuración del gráfico
plt.figure(figsize=(12, 10))
ax = plt.gca()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Colores
colores = {
    'datos': '#3498db',
    'analisis': '#2ecc71',
    'recomendaciones': '#e74c3c',
    'interfaz': '#9b59b6',
    'usuario': '#f39c12',
    'fondo': '#ecf0f1',
    'flecha': '#34495e'
}

# Crear cajas para los módulos
modulos = {
    'fuentes_datos': patches.Rectangle((0.5, 8), 3, 1, facecolor=colores['datos'], alpha=0.8, edgecolor='black'),
    'obtencion_datos': patches.Rectangle((4, 8), 3, 1, facecolor=colores['datos'], alpha=0.8, edgecolor='black'),
    'almacenamiento': patches.Rectangle((7.5, 8), 2, 1, facecolor=colores['datos'], alpha=0.8, edgecolor='black'),
    'usuario': patches.Rectangle((0.5, 5), 2, 1, facecolor=colores['usuario'], alpha=0.8, edgecolor='black'),
    'controlador': patches.Rectangle((3, 5), 2, 1, facecolor=colores['interfaz'], alpha=0.8, edgecolor='black'),
    'analisis': patches.Rectangle((5.5, 5), 2, 1, facecolor=colores['analisis'], alpha=0.8, edgecolor='black'),
    'recomendaciones': patches.Rectangle((8, 5), 2, 1, facecolor=colores['recomendaciones'], alpha=0.8, edgecolor='black'
    ),
    'interfaz': patches.Rectangle((3, 2), 5, 1, facecolor=colores['interfaz'], alpha=0.8, edgecolor='black')
}

# Añadir módulos al gráfico
for modulo in modulos.values():
    ax.add_patch(modulo)

# Añadir texto a los módulos
plt.text(2, 8.5, 'Fuentes de Datos\nExternas', ha='center', va='center', fontsize=10)
plt.text(5.5, 8.5, 'Módulo de\nObtención de Datos', ha='center', va='center', fontsize=10)
plt.text(8.5, 8.5, 'Almacenamiento\nLocal', ha='center', va='center', fontsize=10)
plt.text(1.5, 5.5, 'Entrada del\nUsuario', ha='center', va='center', fontsize=10)
plt.text(4, 5.5, 'Controlador', ha='center', va='center', fontsize=10)
plt.text(6.5, 5.5, 'Módulo de\nAnálisis', ha='center', va='center', fontsize=10)
plt.text(9, 5.5, 'Módulo de\nRecomendaciones', ha='center', va='center', fontsize=10)
plt.text(5.5, 2.5, 'Interfaz de Usuario', ha='center', va='center', fontsize=10)

# Dibujar flechas
def dibujar_flecha(inicio, fin, color=colores['flecha']):
    ax.annotate('', xy=fin, xytext=inicio,
                arrowprops=dict(facecolor=color, shrink=0.05, width=1.5, headwidth=8))

# Flechas horizontales
dibujar_flecha((3.5, 8.5), (4, 8.5))
dibujar_flecha((7, 8.5), (7.5, 8.5))
dibujar_flecha((2.5, 5.5), (3, 5.5))
dibujar_flecha((5, 5.5), (5.5, 5.5))
dibujar_flecha((7.5, 5.5), (8, 5.5))

# Flechas verticales
dibujar_flecha((5.5, 8), (5.5, 6))
dibujar_flecha((9, 5), (9, 3))
dibujar_flecha((9, 3), (8, 2.5))

# Flecha de retroalimentación
plt.arrow(3, 2.5, -0.5, 0, head_width=0.1, head_length=0.1, fc=colores['flecha'], ec=colores['flecha'], linestyle='dashed')

# Título
plt.title('Diagrama de Flujo de Datos - Software de Trading en Futuros', fontsize=14, pad=20)

# Guardar el diagrama
plt.savefig('/home/ubuntu/trading_futures/diagrama_flujo.png', dpi=300, bbox_inches='tight')
plt.close()

print("Diagrama de flujo generado correctamente en: /home/ubuntu/trading_futures/diagrama_flujo.png")
