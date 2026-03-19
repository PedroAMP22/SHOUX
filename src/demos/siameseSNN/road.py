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

from classes import SiameseSNN, AudioMNISTEvalDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "AudioMNIST_split"
BACKBONE_PATH = "src/demos/siameseSNN/models/snn_pop_audio_explainable.pth" 
SIAMESE_SAVE_PATH = "./src/demos/siameseSNN/models/snn_siamese_model.pt"

NUM_SAMPLES = 50       
NUM_SEGMENTS = 16      
TARGET_LEN = 8000
METHODS =["SNN Siamese", "MFCC", "DWT", "Pearson"]


def apply_vad_and_align(waveform, threshold_ratio=0.05, target_len=TARGET_LEN):
    if waveform.dim() == 1: waveform = waveform.unsqueeze(0)
    abs_wave = waveform.abs().squeeze()
    if abs_wave.dim() == 0 or abs_wave.max() == 0: return torch.zeros((1, target_len))
    threshold = abs_wave.max() * threshold_ratio
    active_indices = torch.where(abs_wave > threshold)[0]
    if len(active_indices) == 0: return torch.zeros((1, target_len))
    trimmed = waveform[:, active_indices[0].item():active_indices[-1].item()+1]
    if trimmed.size(1) > target_len: return trimmed[:, :target_len]
    return F.pad(trimmed, (0, target_len - trimmed.size(1)))

def extract_mfcc(waveform):
    mfcc_tr = torchaudio.transforms.MFCC(
        sample_rate=8000, n_mfcc=13,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
    ).to(waveform.device)
    return mfcc_tr(waveform).contiguous().view(1, -1) # [1, features]

def extract_dwt(waveform):
    wave_np = waveform.squeeze().cpu().numpy()
    coeffs = pywt.wavedec(wave_np, 'db4', level=4)
    features = np.concatenate([[np.mean(c), np.std(c), np.sum(c**2)] for c in coeffs])
    feat_tensor = torch.tensor(features / (np.linalg.norm(features) + 1e-9), dtype=torch.float32)
    return feat_tensor.unsqueeze(0).to(waveform.device) # [1, features]

def get_representation(model, waveform, method):
    if method == "SNN Siamese":
        emb, _ = model.get_embedding(waveform)
        return emb
    elif method == "MFCC":
        return extract_mfcc(waveform)
    elif method == "DWT":
        return extract_dwt(waveform)
    elif method == "Pearson":
        wave_c = waveform - waveform.mean()
        return wave_c.view(1, -1)

def get_distance(rep1, rep2, method):
    if method in ["SNN Siamese", "DWT"]:
        return torch.cdist(rep1, rep2, p=2).item()
    elif method in["MFCC", "Pearson"]:
        return (1 - F.cosine_similarity(rep1, rep2, dim=1)).item()

def calculate_importance(model, waveform, method, num_segments=16):
    orig_rep = get_representation(model, waveform, method)
    segment_length = TARGET_LEN // num_segments
    importances =[]
    
    for i in range(num_segments):
        occluded_wave = waveform.clone()
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        occluded_wave[..., start_idx:end_idx] = 0.0
        
        occ_rep = get_representation(model, occluded_wave, method)
        dist = get_distance(orig_rep, occ_rep, method)
        importances.append(dist)
        
    return np.array(importances)

def apply_masking(waveform, importances, mask_percentage, mode="MoRF"):
    num_segments = len(importances)
    num_to_mask = int(num_segments * mask_percentage)
    if num_to_mask == 0: return waveform
        
    if mode == "MoRF": target_indices = np.argsort(importances)[::-1][:num_to_mask]
    elif mode == "LeRF": target_indices = np.argsort(importances)[:num_to_mask]
        
    masked_wave = waveform.clone()
    segment_length = TARGET_LEN // num_segments
    for idx in target_indices:
        start_idx = idx * segment_length
        end_idx = start_idx + segment_length
        masked_wave[..., start_idx:end_idx] = 0.0
        
    return masked_wave

if __name__ == "__main__":
    model = SiameseSNN(BACKBONE_PATH).to(device)
    ckpt = torch.load(SIAMESE_SAVE_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state'] if 'model_state' in ckpt else ckpt, strict=True)
    model.eval() 

    test_ds = AudioMNISTEvalDataset(f"{DATA_ROOT}/test")
    mask_percentages =[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    results = {m: {"MoRF": {p:[] for p in mask_percentages}, 
                   "LeRF": {p:[] for p in mask_percentages}} for m in METHODS}
    
    print(f"\nIniciando ROAD Comparativo sobre {NUM_SAMPLES} audios...")
    np.random.seed(42)
    indices = np.random.choice(len(test_ds), NUM_SAMPLES, replace=False)
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="Evaluando"):
            wave_raw, _, _ = test_ds[idx]
            wave_aligned = apply_vad_and_align(wave_raw)
            
            if wave_aligned.dim() == 2: wave_input = wave_aligned.unsqueeze(0)
            else: wave_input = wave_aligned.unsqueeze(0).unsqueeze(0)
            if wave_input.dim() == 4: wave_input = wave_input.squeeze(2)
            
            wave_input = wave_input.to(device)

            for m in METHODS:
                orig_rep = get_representation(model, wave_input, m)
                importances = calculate_importance(model, wave_input, m, NUM_SEGMENTS)
                
                for p in mask_percentages:
                    for mode in ["MoRF", "LeRF"]:
                        masked_wave = apply_masking(wave_input, importances, p, mode)
                        new_rep = get_representation(model, masked_wave, m)
                        error = get_distance(orig_rep, new_rep, m)
                        
                        results[m][mode][p].append(error)


    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle('ROAD Benchmark: Explanation Fidelity Comparison', fontsize=18, fontweight='bold', y=1.02)

    colors = {'SNN Siamese': '#d62728', 'MFCC': '#ff7f0e', 'DWT': '#1f77b4', 'Pearson': '#9467bd'}
    markers = {'SNN Siamese': 'o', 'MFCC': 's', 'DWT': '^', 'Pearson': 'D'}

    x_vals =[p * 100 for p in mask_percentages]

    for m in METHODS:
        y_morf = [np.mean(results[m]["MoRF"][p]) for p in mask_percentages]
        linewidth = 3 if m == 'SNN Siamese' else 2
        ax1.plot(x_vals, y_morf, label=m, color=colors[m], marker=markers[m], linewidth=linewidth, markersize=8)

    ax1.set_title('MoRF: Removing Most Relevant Features (Higher is Better)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Percentage of Audio Erased (%)', fontsize=12)
    ax1.set_ylabel('Latent Degradation (Distance to Original)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.7)

    for m in METHODS:
        y_lerf = [np.mean(results[m]["LeRF"][p]) for p in mask_percentages]
        linewidth = 3 if m == 'SNN Siamese' else 2
        ax2.plot(x_vals, y_lerf, label=m, color=colors[m], marker=markers[m], linewidth=linewidth, markersize=8)

    ax2.set_title('LeRF: Removing Least Relevant Features (Lower is Better)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Percentage of Audio Erased (%)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("road_comparative_all.png", dpi=300, bbox_inches='tight')
    plt.show()