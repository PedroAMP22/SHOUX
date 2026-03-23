import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

DB_CACHE_PATH = "db_cache.pt"
K_MAX = 20
k_values = list(range(1, K_MAX + 1))

if not os.path.exists(DB_CACHE_PATH):
    raise FileNotFoundError("No db_cache.pt")

db = torch.load(DB_CACHE_PATH, weights_only=False)

avg_accuracy_k =[]
avg_factual_rmse_k =[]

snn_embs = db['snn']
all_dists = torch.cdist(snn_embs, snn_embs, p=2)

for k in tqdm(k_values, desc="Evaluatin Ks"):
    acc_list = []
    rmse_list =[]
    
    for i in range(len(db['filenames'])):
        q_label = db['labels'][i].item()
        q_gold = db['gold_vector'][i].unsqueeze(0)
        
        dists_i = all_dists[i].clone()
        dists_i[i] = float('inf')
        
        topk_idx = torch.topk(dists_i, k, largest=False)[1]
        hits = (db['labels'][topk_idx] == q_label).float().mean().item()
        acc_list.append(hits)
        
        mask_factual = (db['labels'] == q_label)
        mask_factual[i] = False 
        
        dists_f = dists_i[mask_factual]
        if len(dists_f) > 0:
            actual_k = min(k, len(dists_f))
            topk_f_idx = torch.topk(dists_f, actual_k, largest=False)[1]
            
            neighbors_gold = db['gold_vector'][mask_factual][topk_f_idx]
            rmse = torch.sqrt(torch.mean((q_gold - neighbors_gold)**2, dim=1)).mean().item()
            rmse_list.append(rmse)
            
    avg_accuracy_k.append(np.mean(acc_list) * 100)
    avg_factual_rmse_k.append(np.mean(rmse_list))

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax1 = plt.subplots(figsize=(10, 6))

color_acc = '#1f77b4'
color_rmse = '#d62728'

ax1.set_xlabel('Number of Factuals (K)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy in Neigbors(%)', color=color_acc, fontsize=12, fontweight='bold')
line1 = ax1.plot(k_values, avg_accuracy_k, color=color_acc, marker='o', linewidth=2.5, label='Accruacy')
ax1.tick_params(axis='y', labelcolor=color_acc)
ax1.set_xticks(k_values)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()  
ax2.set_ylabel('Factual RMSE', color=color_rmse, fontsize=12, fontweight='bold')
line2 = ax2.plot(k_values, avg_factual_rmse_k, color=color_rmse, marker='s', linewidth=2.5, linestyle='--', label='Factual RMSE')
ax2.tick_params(axis='y', labelcolor=color_rmse)
ax2.grid(False)

lines = line1 + line2
labels =[l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', fontsize=11)

plt.title('SNN Siamese: K Threshold', fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig("optimal_k_threshold.png", dpi=300, bbox_inches='tight')
plt.show()