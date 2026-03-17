import matplotlib.pyplot as plt
import numpy as np
import re

FILE_NAME = "resultados.txt" 

def parse_txt_data(filename):
    data = {1: {}, 3: {}, 5: {}}
    current_k = None
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if "Results K=" in line:
                    current_k = int(re.search(r'K=(\d+)', line).group(1))
                    continue
                if "Method" in line or "---" in line or "|" not in line:
                    continue
                
                parts =[p.strip() for p in line.split("|")]
                if len(parts) >= 7:
                    method_name = parts[0]
                    f_rmse = float(parts[3])
                    cf_rmse = float(parts[4])
                    f_div = float(parts[5])
                    cf_div = float(parts[6])
                    
                    data[current_k][method_name] = {
                        'f_rmse': f_rmse, 'cf_rmse': cf_rmse,
                        'f_div': f_div, 'cf_div': cf_div
                    }
        return data
    except FileNotFoundError:
        print(f"Error: No se encontró {filename}")
        return None

raw_data = parse_txt_data(FILE_NAME)

if raw_data:
    methods =['SNN Siamese', 'Van Rossum', 'MFCC', 'DWT', 'Pearson']
    colors =['#d62728', '#2ca02c', '#ff7f0e', '#1f77b4', '#9467bd'] 

    K_PLOT = 5
    
    f_rmse = [raw_data[K_PLOT][m]['f_rmse'] for m in methods]
    cf_rmse = [raw_data[K_PLOT][m]['cf_rmse'] for m in methods]
    f_div = [raw_data[K_PLOT][m]['f_div'] for m in methods]
    cf_div = [raw_data[K_PLOT][m]['cf_div'] for m in methods]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f'xAI Latent Space Geometry (K={K_PLOT})', fontsize=22, fontweight='bold', y=1.05)

    x = np.arange(len(methods))
    width = 0.35

    # Graph 1: Explanation Fidelity (RMSE to Gold Standard)
    ax1 = axes[0]
    rects1 = ax1.bar(x - width/2, f_rmse, width, label='Factual RMSE (↓ Better)', color='#1f77b4', edgecolor='black')
    rects2 = ax1.bar(x + width/2, cf_rmse, width, label='Counterfactual RMSE (↑ Better)', color='#d62728', edgecolor='black')

    ax1.set_title('Explanation Fidelity (Distance to Gold Standard)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('RMSE (Wav2Vec Space)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=30, ha='right', fontsize=12)
    ax1.legend(fontsize=12)
    

    ax1.set_ylim(min(f_rmse) - 0.01, max(cf_rmse) + 0.01)

    # 2nd graph: Diversity (Intra-class vs Inter-class)
    ax2 = axes[1]
    rects3 = ax2.bar(x - width/2, f_div, width, label='Factual Diversity (↓ Better)', color='#2ca02c', edgecolor='black')
    rects4 = ax2.bar(x + width/2, cf_div, width, label='Counterfactual Diversity (↑ Better)', color='#9467bd', edgecolor='black')

    ax2.set_title('Latent Topology (Intra-class vs Inter-class)', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Diversity Score (Pairwise Distance)', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=30, ha='right', fontsize=12)
    ax2.legend(fontsize=12)

    ax2.set_ylim(min(f_div) - 0.5, max(cf_div) + 0.5)

    plt.tight_layout()
    plt.savefig("xai_geometry_metrics.png", dpi=300, bbox_inches='tight')
    plt.show()