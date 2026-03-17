import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from classes import SiameseSNN, AudioMNISTEvalDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "AudioMNIST_split"
BACKBONE_PATH = "src/demos/siameseSNN/models/snn_pop_audio_explainable.pth" 
SIAMESE_SAVE_PATH = "./src/demos/siameseSNN/models/snn_siamese_model.pt"

NUM_SAMPLES = 50       
NUM_SEGMENTS = 16      
TARGET_LEN = 8000

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

def calculate_importance_by_occlusion(model, waveform, num_segments=16):
    model.eval()
    with torch.no_grad():
        orig_emb, _ = model.get_embedding(waveform.to(device))
        segment_length = TARGET_LEN // num_segments
        importances =[]
        
        for i in range(num_segments):
            occluded_wave = waveform.clone()
            start_idx = i * segment_length
            end_idx = start_idx + segment_length
            
            occluded_wave[..., start_idx:end_idx] = 0.0
            
            occ_emb, _ = model.get_embedding(occluded_wave.to(device))
            dist = torch.cdist(orig_emb, occ_emb, p=2).item()
            importances.append(dist)
            
    return np.array(importances)

def apply_road_masking(waveform, importances, mask_percentage, mode="MoRF"):
    num_segments = len(importances)
    num_to_mask = int(num_segments * mask_percentage)
    
    if num_to_mask == 0:
        return waveform
        
    if mode == "MoRF":
        target_indices = np.argsort(importances)[::-1][:num_to_mask]
    elif mode == "LeRF":
        target_indices = np.argsort(importances)[:num_to_mask]
    else:
        target_indices = np.random.choice(num_segments, num_to_mask, replace=False)
        
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
    state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    
    model.eval() 

    test_ds = AudioMNISTEvalDataset(f"{DATA_ROOT}/test")
    
    mask_percentages = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    results = {
        "MoRF": {p: [] for p in mask_percentages},
        "LeRF": {p: [] for p in mask_percentages},
        "Random": {p: [] for p in mask_percentages}
    }
        
    np.random.seed(42)
    indices = np.random.choice(len(test_ds), NUM_SAMPLES, replace=False)
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="Evaluando ROAD"):
            wave_raw, label, fname = test_ds[idx]
            
            wave_aligned = apply_vad_and_align(wave_raw)
            

            if wave_aligned.dim() == 2:
                wave_input = wave_aligned.unsqueeze(0) 
            else:
                wave_input = wave_aligned.unsqueeze(0).unsqueeze(0)
            

            if wave_input.dim() == 4:
                wave_input = wave_input.squeeze(2) 
            
            # Original emb
            orig_emb, _ = model.get_embedding(wave_input.to(device))
                
            # Calculate importance
            importances = calculate_importance_by_occlusion(model, wave_input, NUM_SEGMENTS)
            
            # ROAD
            for p in mask_percentages:
                for mode in ["MoRF", "LeRF", "Random"]:
                    masked_wave = apply_road_masking(wave_input, importances, p, mode)
                    
                    # new emb
                    new_emb, _ = model.get_embedding(masked_wave.to(device))
                        
                    # l2 distance
                    error = torch.cdist(orig_emb, new_emb, p=2).item()
                    results[mode][p].append(error)

    # grpahs
    print("\n" + "="*60)
    print(f"{'Erased':<20} | {'MoRF Error':<12} | {'LeRF Error':<12} | {'Random Error'}")
    print("="*60)
    
    morf_means, lerf_means, rand_means = [], [], []
    
    for p in mask_percentages:
        m_err = np.mean(results["MoRF"][p])
        l_err = np.mean(results["LeRF"][p])
        r_err = np.mean(results["Random"][p])
        morf_means.append(m_err)
        lerf_means.append(l_err)
        rand_means.append(r_err)
        print(f"{int(p*100):>17}% | {m_err:>12.4f} | {l_err:>12.4f} | {r_err:>12.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot([p*100 for p in mask_percentages], morf_means, marker='o', color='red', linewidth=2, label='MoRF Delete Important')
    plt.plot([p*100 for p in mask_percentages], rand_means, marker='s', color='gray', linewidth=2, linestyle='--', label='Random')
    plt.plot([p*100 for p in mask_percentages], lerf_means, marker='^', color='green', linewidth=2, label='LeRF Delete Noise)')
    plt.title("ROAD", fontsize=16, fontweight='bold')
    plt.xlabel("Audio erased(%)", fontsize=12)
    plt.ylabel("Distance Actual-Original", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("road_evaluation.png", dpi=300)
    plt.show()