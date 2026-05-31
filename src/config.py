#===========================
#  AJUSTES DE PREPROCESAMIENTO DE AUDIO
#===========================
SR = 16000
TOP_DB = 20
MONO = True

SEED = 18022025

caracteristicas_vocales = ['Age', 'Sex', 'ATRI', 'rAPQ', 'rPPQ', 'CHNR', 'Hnr']
caracteristicas_frases = ['Age', 'Sex', 'ShimmerDb', 'ATRI', 'Hnr', 'CHNR', 'rPPQ']
caracteristicas_globales = ['Age', 'Sex', 'ShimmerDb', 'ATRI', 'Hnr', 'CHNR', 'rPPQ']

#===========================
#  DISEÑO DE VISUALIZACIÓN
#===========================

import matplotlib.pyplot as plt

PALETTE = {
    "bg":       "#FFFFFF",   # Fondo blanco puro
    "panel":    "#F8F9FA",   # Gris muy claro para paneles/tablas
    "accent1":  "#2B5B84",   # Azul oscuro elegante
    "accent2":  "#4CAF50",   # Verde vibrante pero legible
    "accent3":  "#E67E22",   # Naranja (contraste)
    "accent4":  "#8E44AD",   # Púrpura suave
    "warn":     "#E74C3C",   # Rojo para errores/overfitting
    "text":     "#2C3E50",   # Texto oscuro casi negro
    "subtext":  "#7F8C8D",   # Gris para textos secundarios y ejes
}

def apply_style():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["bg"],
        "axes.edgecolor":    PALETTE["subtext"],
        "axes.labelcolor":   PALETTE["text"],
        "axes.titlecolor":   PALETTE["text"],
        "axes.grid":         True,
        "grid.color":        "#E0E0E0", # Grid gris muy sutil
        "grid.linestyle":    "--",
        "grid.alpha":        0.7,
        "xtick.color":       PALETTE["subtext"],
        "ytick.color":       PALETTE["subtext"],
        "text.color":        PALETTE["text"],
        "legend.facecolor":  PALETTE["bg"],
        "legend.edgecolor":  PALETTE["subtext"],
        "font.family":       "DejaVu Sans",
        "font.size":         10,
    })
