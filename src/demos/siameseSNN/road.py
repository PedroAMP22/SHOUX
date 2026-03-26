import torch
import torch.nn.functional as F
import numpy as np
import torchaudio
import pywt
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if directorio_raiz not in sys.path:
    sys.path.append(directorio_raiz)

from classes import SiameseSNN

device =  "cpu"
DATA_ROOT = "AudioMNIST_split"
BACKBONE_PATH = "src/demos/siameseSNN/models/snn_pop_audio_explainable.pth" 
SIAMESE_SAVE_PATH = "./src/demos/siameseSNN/models/snn_siamese_model.pt"
DB_CACHE_PATH = "db_cache.pt"

NUM_SAMPLES = 100
NUM_SEGMENTS = 16      
TARGET_LEN = 8000
METHODS =["SNN Siamese", "MFCC", "DWT", "Pearson"]
K_VALUES = [1, 3, 5]


def extract_mfcc(waveform):
    mfcc_tr = torchaudio.transforms.MFCC(
        sample_rate=8000, n_mfcc=13,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
    ).to(waveform.device)
    return mfcc_tr(waveform).contiguous().view(1, -1)

def extract_dwt(waveform):
    wave_np = waveform.squeeze().cpu().numpy()
    coeffs = pywt.wavedec(wave_np, 'db4', level=4)
    features = np.concatenate([[np.mean(c), np.std(c), np.sum(c**2)] for c in coeffs])
    feat_tensor = torch.tensor(features / (np.linalg.norm(features) + 1e-9), dtype=torch.float32)
    return feat_tensor.unsqueeze(0).to(waveform.device)

def get_representation(model, waveform, method):
    if method == "SNN Siamese":
        emb, _ = model.get_embedding(waveform)
        return emb
    elif method == "MFCC":
        return extract_mfcc(waveform)
    elif method == "DWT":
        return extract_dwt(waveform)
    elif method == "Pearson":
        return (waveform - waveform.mean()).view(1, -1)

def get_distance(rep1, rep2, method):
    if method in ["SNN Siamese", "DWT"]:
        return torch.cdist(rep1, rep2, p=2).item()
    else:
        return (1 - F.cosine_similarity(rep1, rep2, dim=1)).item()
 
def calculate_explanation_importance(model, q_rep, exp_wave, method):
    orig_exp_rep = get_representation(model, exp_wave, method)
    orig_dist = get_distance(q_rep, orig_exp_rep, method)
    
    segment_length = TARGET_LEN // NUM_SEGMENTS
    importances =[]
    for i in range(NUM_SEGMENTS):
        occ_wave = exp_wave.clone()
        occ_wave[..., i*segment_length : (i+1)*segment_length] = 0.0
        occ_rep = get_representation(model, occ_wave, method)
        importances.append(get_distance(q_rep, occ_rep, method) - orig_dist)
    return np.array(importances)

def apply_masking(waveform, importances, mask_percentage, mode):
    num_to_mask = int(NUM_SEGMENTS * mask_percentage)
    if num_to_mask == 0: return waveform
    if mode == "MoRF": idxs = np.argsort(importances)[::-1][:num_to_mask]
    else: idxs = np.argsort(importances)[:num_to_mask]
    
    masked = waveform.clone()
    seg_len = TARGET_LEN // NUM_SEGMENTS
    for i in idxs: masked[..., i*seg_len : (i+1)*seg_len] = 0.0
    return masked

if __name__ == "__main__":
    print("Cargando modelo y base de datos...")
    model = SiameseSNN(BACKBONE_PATH).to(device)
    ckpt = torch.load(SIAMESE_SAVE_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state'] if 'model_state' in ckpt else ckpt, strict=True)
    model.eval() 

    if not os.path.exists(DB_CACHE_PATH):
        raise FileNotFoundError("Missing db_cache.pt")
    db = torch.load(DB_CACHE_PATH, weights_only=False)

    mask_percentages =[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    results = {k: {m: {
        "Factual": {"MoRF": {p: [] for p in mask_percentages}, "LeRF": {p:[] for p in mask_percentages}},
        "Counterfactual": {"MoRF": {p:[] for p in mask_percentages}, "LeRF": {p:[] for p in mask_percentages}}
    } for m in METHODS} for k in K_VALUES}

    np.random.seed(42)
    indices = np.random.choice(len(db['filenames']), NUM_SAMPLES, replace=False)
    
    max_k = max(K_VALUES) 

    with torch.no_grad():
        for idx in tqdm(indices, desc="Evaluando ROAD"):
            q_label = db['labels'][idx].item()
            q_wave = db['raw_c'][idx].view(1, 1, -1).to(device)

            mask_self = torch.arange(len(db['filenames'])) != idx
            mask_factual = mask_self & (db['labels'] == q_label)
            mask_cf = db['labels'] != q_label

            for m in METHODS:
                q_rep = get_representation(model, q_wave, m)
                
                if m == "SNN Siamese": d_all = torch.cdist(q_rep, db['snn'], p=2).squeeze(0)
                elif m == "MFCC": d_all = torch.cdist(q_rep, db['mfcc'], p=2).squeeze(0)
                elif m == "DWT": d_all = torch.cdist(q_rep, db['dwt'], p=2).squeeze(0)
                elif m == "Pearson": d_all = 1 - F.cosine_similarity(q_rep.cpu(), db['raw_c'].squeeze(1)).to(device)

                top_f_idx = torch.where(mask_factual)[0][torch.topk(d_all[mask_factual], max_k, largest=False)[1]]
                top_cf_idx = torch.where(mask_cf)[0][torch.topk(d_all[mask_cf], max_k, largest=False)[1]]

                for rank, f_idx in enumerate(top_f_idx):
                    exp_wave = db['raw_c'][f_idx].view(1, 1, -1).to(device)
                    importances = calculate_explanation_importance(model, q_rep, exp_wave, m)
                    orig_dist = get_distance(q_rep, get_representation(model, exp_wave, m), m)
                    
                    for p in mask_percentages:
                        for mode in["MoRF", "LeRF"]:
                            masked_wave = apply_masking(exp_wave, importances, p, mode)
                            new_dist = get_distance(q_rep, get_representation(model, masked_wave, m), m)
                            degradation = new_dist - orig_dist
                            
                            for k in K_VALUES:
                                if rank < k:
                                    results[k][m]["Factual"][mode][p].append(degradation)

                for rank, cf_idx in enumerate(top_cf_idx):
                    exp_wave = db['raw_c'][cf_idx].view(1, 1, -1).to(device)
                    importances = calculate_explanation_importance(model, q_rep, exp_wave, m)
                    orig_dist = get_distance(q_rep, get_representation(model, exp_wave, m), m)
                    
                    for p in mask_percentages:
                        for mode in ["MoRF", "LeRF"]:
                            masked_wave = apply_masking(exp_wave, importances, p, mode)
                            new_dist = get_distance(q_rep, get_representation(model, masked_wave, m), m)
                            degradation = new_dist - orig_dist
                            
                            for k in K_VALUES:
                                if rank < k:
                                    results[k][m]["Counterfactual"][mode][p].append(degradation)

    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {'SNN Siamese': '#d62728', 'MFCC': '#ff7f0e', 'DWT': '#1f77b4', 'Pearson': '#9467bd'}
    markers = {'SNN Siamese': 'o', 'MFCC': 's', 'DWT': '^', 'Pearson': 'D'}
    x_vals = [p * 100 for p in mask_percentages]

    titles =[
        ("Factual", "MoRF", "Factual: MoRF (Higher is Better)"),
        ("Factual", "LeRF", "Factual: LeRF (Lower is Better)"),
        ("Counterfactual", "MoRF", "Counterfactual: MoRF (Higher is Better)"),
        ("Counterfactual", "LeRF", "Counterfactual: LeRF (Lower is Better)")
    ]

    for k in K_VALUES:
        print(f"\n{'='*80}")
        print(f" K = {k}")
        print(f"{'='*80}")
        print(f"{'Method':<15} | {'Factual MoRF':<18} | {'Factual LeRF':<18} | {'CF MoRF':<15} | {'CF LeRF'}")
        print("-" * 80)
        
        for m in METHODS:
            f_morf_50 = np.mean(results[k][m]["Factual"]["MoRF"][0.5])
            f_lerf_50 = np.mean(results[k][m]["Factual"]["LeRF"][0.5])
            cf_morf_50 = np.mean(results[k][m]["Counterfactual"]["MoRF"][0.5])
            cf_lerf_50 = np.mean(results[k][m]["Counterfactual"]["LeRF"][0.5])
            print(f"{m:<15} | {f_morf_50:>18.4f} | {f_lerf_50:>18.4f} | {cf_morf_50:>15.4f} | {cf_lerf_50:>12.4f}")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
        fig.suptitle(f'Explanation Quality via ROAD (Avg over Top-{k} Explanations)', fontsize=18, fontweight='bold', y=0.96)

        for i, (exp_type, mode, title) in enumerate(titles):
            ax = axes[i//2, i%2]
            for m in METHODS:
                y = [np.mean(results[k][m][exp_type][mode][p]) for p in mask_percentages]
                lw = 3 if m == 'SNN Siamese' else 2
                ax.plot(x_vals, y, label=m, color=colors[m], marker=markers[m], linewidth=lw, markersize=8)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.7)
            if i >= 2: ax.set_xlabel("Percentage of Explanation Erased (%)", fontsize=12)
            if i % 2 == 0: ax.set_ylabel("$\Delta$ Distance to Query", fontsize=12)
            if i == 0: ax.legend(fontsize=11)

        plt.tight_layout()
        filename = f"road_explanation_k{k}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"-> Gráfica guardada: {filename}")
        
    plt.show()