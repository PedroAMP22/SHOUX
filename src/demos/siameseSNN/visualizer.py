import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
DB_CACHE_PATH = "db_cache.pt"
NUM_SAMPLES_TO_PLOT = 1000 # Usamos 1000 muestras para que la gráfica sea clara y no un borrón

# ==========================================
# 2. CARGAR BASE DE DATOS
# ==========================================
if not os.path.exists(DB_CACHE_PATH):
    raise FileNotFoundError(f"No se encontró {DB_CACHE_PATH}. Ejecuta primero el script principal para generarla.")

print("Cargando base de datos desde caché...")
db = torch.load(DB_CACHE_PATH, weights_only=False)

# Extraer datos
labels = db['labels'].numpy()
total_samples = len(labels)

# Seleccionar un subconjunto aleatorio para la visualización (t-SNE es lento con muchos datos)
np.random.seed(42) # Semilla fija para reproducibilidad
indices = np.random.choice(total_samples, min(NUM_SAMPLES_TO_PLOT, total_samples), replace=False)

labels_sub = labels[indices]

# Diccionario con las representaciones vectoriales aplanar a 2D
features_dict = {
    "SNN Siamese (Ours)": db['snn'][indices].numpy(),
    "MFCC (SOTA)": db['mfcc'][indices].numpy(),
    "DWT": db['dwt'][indices].numpy(),
    "Pearson (Raw Audio)": db['raw_c'][indices].squeeze(1).numpy() # Quitamos la dimensión extra si la hay
}

# ==========================================
# 3. APLICAR t-SNE
# ==========================================
tsne_results = {}
print("\nCalculando t-SNE para cada espacio latente (esto puede tardar un par de minutos)...")

for name, feats in features_dict.items():
    print(f" -> Procesando {name}...")
    # t-SNE con parámetros estándar para clustering
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, init='pca')
    
    # Aplanar características si tienen más de 2 dimensiones
    if len(feats.shape) > 2:
        feats = feats.reshape(feats.shape[0], -1)
        
    tsne_results[name] = tsne.fit_transform(feats)

# Paleta de colores global
cmap = plt.get_cmap('tab10')
legend_labels = [f"Digit {j}" for j in range(10)]

# ==========================================
# 4. GUARDAR GRÁFICAS INDIVIDUALES
# ==========================================
print("\nGenerando y guardando gráficas individuales...")

for name, embedding in tsne_results.items():
    fig_indiv, ax_indiv = plt.subplots(figsize=(8, 6))
    
    scatter_indiv = ax_indiv.scatter(embedding[:, 0], embedding[:, 1], 
                                     c=labels_sub, cmap=cmap, alpha=0.8, s=20, edgecolors='white', linewidth=0.3)
    
    ax_indiv.set_title(name, fontsize=18, fontweight='bold')
    ax_indiv.set_xticks([])
    ax_indiv.set_yticks([])
    ax_indiv.set_facecolor('#F9F9F9')
    
    for spine in ax_indiv.spines.values():
        spine.set_color('#DDDDDD')
        
    # Leyenda para la gráfica individual (a la derecha)
    handles, _ = scatter_indiv.legend_elements(prop="colors", alpha=1)
    ax_indiv.legend(handles, legend_labels, loc='center left', bbox_to_anchor=(1.02, 0.5), title="Clases", fontsize=10)
    
    # Crear un nombre de archivo seguro (ej. "SNN Siamese (Ours)" -> "tsne_snn_siamese_ours.png")
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    indiv_filename = f"tsne_{safe_name}.png"
    
    plt.tight_layout()
    fig_indiv.savefig(indiv_filename, dpi=300, bbox_inches='tight')
    plt.close(fig_indiv) # Cerramos la figura para no saturar la memoria
    print(f" -> Guardada: {indiv_filename}")

# ==========================================
# 5. VISUALIZACIÓN COMBINADA (2x2)
# ==========================================
print("\nGenerando gráfica combinada (2x2)...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, (name, embedding) in enumerate(tsne_results.items()):
    ax = axes[i]
    
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                         c=labels_sub, cmap=cmap, alpha=0.8, s=20, edgecolors='white', linewidth=0.3)
    
    ax.set_title(name, fontsize=16, fontweight='bold')
    ax.set_xticks([]) 
    ax.set_yticks([])
    ax.set_facecolor('#F9F9F9')
    for spine in ax.spines.values():
        spine.set_color('#DDDDDD')
# Leyenda común (a la derecha de la figura combinada)
handles, _ = scatter.legend_elements(prop="colors", alpha=1)
fig.legend(handles, legend_labels, loc='center right', title="Clases", fontsize=12)
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Deja espacio a la derecha para la leyenda
combined_filename = "tsne_combined.png"
fig.savefig(combined_filename, dpi=300, bbox_inches='tight')
plt.close(fig) # Cerramos la figura para no saturar la memoria
print(f" -> Gráfica combinada guardada: {combined_filename}")