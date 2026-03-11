import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# Config
DB_CACHE_PATH = "db_cache.pt"
NUM_SAMPLES_TO_PLOT = 1000 

# Load databse
if not os.path.exists(DB_CACHE_PATH):
    raise FileNotFoundError(f"Error: Base data cahe file not found")

db = torch.load(DB_CACHE_PATH, weights_only=False)

labels = db['labels'].numpy()
total_samples = len(labels)

np.random.seed(42)
indices = np.random.choice(total_samples, min(NUM_SAMPLES_TO_PLOT, total_samples), replace=False)

labels_sub = labels[indices]

features_dict = {
    "SNN Siamese (Ours)": db['snn'][indices].numpy(),
    "MFCC (SOTA)": db['mfcc'][indices].numpy(),
    "DWT": db['dwt'][indices].numpy(),
    "Pearson (Raw Audio)": db['raw_c'][indices].squeeze(1).numpy() 
}

# t-SNE
tsne_results = {}
print("Calculating t-SNE embeddings...")
for name, feats in features_dict.items():
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, init='pca')
    
    if len(feats.shape) > 2:
        feats = feats.reshape(feats.shape[0], -1)
        
    tsne_results[name] = tsne.fit_transform(feats)

cmap = plt.get_cmap('tab10')
legend_labels = [f"Digit {j}" for j in range(10)]

# Indv Plots
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
        
    handles, _ = scatter_indiv.legend_elements(prop="colors", alpha=1)
    ax_indiv.legend(handles, legend_labels, loc='center left', bbox_to_anchor=(1.02, 0.5), title="Clases", fontsize=10)
    
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    indiv_filename = f"tsne_{safe_name}.png"
    
    plt.tight_layout()
    fig_indiv.savefig(indiv_filename, dpi=300, bbox_inches='tight')
    plt.close(fig_indiv)
    print(f" -> Guardada: {indiv_filename}")

# Combined Visual
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

handles, _ = scatter.legend_elements(prop="colors", alpha=1)
fig.legend(handles, legend_labels, loc='center right', title="Clases", fontsize=12)
plt.tight_layout(rect=[0, 0, 0.85, 1]) 
combined_filename = "tsne_combined.png"
fig.savefig(combined_filename, dpi=300, bbox_inches='tight')
plt.close(fig)
